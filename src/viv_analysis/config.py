"""Fixed parameters for both datasets, the GRU and training.

All values are in SI units. Every script reads its constants from here, so a
change in this file changes the whole pipeline. The CFD setup is described in
thesis Ch. 3, the GRU and training settings in Sec. 4.4-4.5 (Table 4.1).
"""

import numpy as np

config = {
    # Bridge deck (thesis Sec. 3.4). D = deck depth [m], B = deck width [m], fn [Hz].
    "bridge_D_ref":       7.42,
    "bridge_B_ref":       25.9,
    "bridge_fn_hz":       0.32,
    # Nondimensional release time t* = t U / D; converted per case as t = t* D / U.
    "bridge_t_star_release": 20.0,
    # Keep every 20th CFD sample: 0.0001 s -> 0.002 s (thesis Sec. 3.5).
    "bridge_downsample":  20,
    # 2500 samples x 0.002 s = 5 s of input history.
    "bridge_seq_len":     2500,
    "bridge_stride_train": 5,
    "bridge_zeta":          0.01,
    # Air density [kg/m^3], mass per unit span [kg/m], damping ratio [-].
    "bridge_rho":           1.225,
    "bridge_mass":          24604.0,

    # Circular cylinder, Leontini et al. benchmark at Re = 200 (thesis Sec. 3.3).
    # D [m], rho [kg/m^3], mass ratio m* [-], damping ratio [-], fn [Hz], dt [s].
    "cylinder200_D_ref":           0.2,
    "cylinder200_rho":             1.0,
    "cylinder200_Re":              200.0,
    "cylinder200_M_star":          10.0,
    "cylinder200_zeta":            0.01,
    "cylinder200_fn":              0.2,
    "cylinder200_dt":              0.005,
    "cylinder200_t_star_release":  40.0,
    "cylinder200_ref_area":        0.2,

    # GRU architecture (thesis Table 4.1).
    "hidden_size":   64,
    "num_layers":    2,
    "dropout":       0.1,

    # Adam learning rate and weight decay; early stopping after `patience` epochs
    # without validation improvement.
    "lr":            1e-3,
    "weight_decay":  1e-5,
    "n_epochs":      100,
    "batch_size":    512,
    "patience":      15,

    # Filled in per dataset by prepare_gru_config().
    "seq_len":       None,
    "stride_train":  None,
    "stride_val":    1,
    "use_ur_context": False,

    "seed":          123,
    "target_col":    "cl",
    # Default inputs. The thesis models use [disp, vel] (acceleration removed, Sec. 5.4).
    "input_cols":    ["disp", "vel", "acc"],
    "motion_type":   "heave",
}

CYLINDER200_ALIASES = frozenset({
    "cylinder200", "cylinder_re200", "cylinder_re_200", "re200",
    "cylinder-re-200",
})

np.random.seed(config["seed"])


def prepare_gru_config(dataset: str, cfg: dict) -> dict:
    """Return a copy of cfg with the dataset-specific sequence settings.

    Cylinder: seq_len 1000 (5 s at 0.005 s), training stride 8.
    Bridge:   seq_len 2500 (5 s at 0.002 s), training stride 5.
    Both use the Ur context feature by default.
    """
    out = cfg.copy()
    ds  = dataset.strip().lower()

    if ds in CYLINDER200_ALIASES:
        out["seq_len"]      = 1000
        out["stride_train"] = 8
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = True

    elif ds == "bridge":
        out["seq_len"]      = 2500
        out["stride_train"] = 5
        out["hidden_size"]  = cfg["hidden_size"]
        out["use_ur_context"] = True

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return out

def cylinder200_U(Ur: float) -> float:
    """Free-stream velocity U = Ur * fn * D [m/s]."""
    return float(Ur) * config["cylinder200_fn"] * config["cylinder200_D_ref"]


def cylinder200_release_time(Ur: float) -> float:
    """Time [s] at which the cylinder is released: t = t*_release * D / U."""
    U = cylinder200_U(Ur)
    return config["cylinder200_t_star_release"] * config["cylinder200_D_ref"] / U


def cylinder200_structural_params() -> dict:
    """Mass, damping and stiffness per unit span of the cylinder.

    m = m* rho pi D^2 / 4,  k = m (2 pi fn)^2,  c = 2 m (2 pi fn) zeta.
    The same m, c, k are used by the Fluent UDF and by the Newmark integrator.
    """
    rho = config["cylinder200_rho"]
    D   = config["cylinder200_D_ref"]
    fn  = config["cylinder200_fn"]

    M_star = config["cylinder200_M_star"]
    zeta   = config["cylinder200_zeta"]
    m = M_star * rho * (np.pi * D**2 / 4.0)
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta

    return {
        "m": m,
        "c": c,
        "k": k,
        "cylinder_mass": m,
        "c_struct": c,
        "k_struct": k,
    }


def bridge_structural_params() -> dict:
    """Mass, damping and stiffness per unit span of the bridge deck (k = m w^2, c = 2 m w zeta)."""
    rho = config['bridge_rho']
    D   = config['bridge_D_ref']
    fn  = config['bridge_fn_hz']
    m   = config['bridge_mass']

    zeta   = config['bridge_zeta']
    omega_n = 2.0 * np.pi * fn
    k = m * omega_n**2
    c = 2.0 * m * omega_n * zeta

    params_bridge = {
            "c": c,
            "k": k,
            "m": m,
        }

    return params_bridge
