"""Read the Fluent monitor files and build one dataframe per dataset (thesis Sec. 3.5, 4.3).

Expected layout (one .out file per case and signal):
    data/cylinder_Re_200/{disp,cl,cd,vel,force}/
    data/Bridge/{disp,cl,cm,vel,force}/
Bridge files are named by wind speed (e.g. disp-16.out) and are converted to
Ur labels. The output has one row per time step with columns
case, step, time, disp, cd (or cm), cl, vel, acc.

The full bridge preprocessing takes about an hour, so its result is cached
in data/cache/ as parquet (see load_bridge_df_cached).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os
import re
from scipy.signal import savgol_filter
from viv_analysis.utils import PROJECT_ROOT, format_ur_label, parse_ur_label
from viv_analysis.config import config, CYLINDER200_ALIASES


DIR = PROJECT_ROOT


CYLINDER200_ROOT     = DIR / "data" / "cylinder_Re_200"
CYLINDER200_DISP_DIR = CYLINDER200_ROOT / "disp"
CYLINDER200_CD_DIR   = CYLINDER200_ROOT / "cd"
CYLINDER200_CL_DIR   = CYLINDER200_ROOT / "cl"
CYLINDER200_VEL_DIR  = CYLINDER200_ROOT / "vel"
CYLINDER200_FY_DIR   = CYLINDER200_ROOT / "force"

BRIDGE_DISP_DIR = DIR / "data" / "Bridge" / "disp"
BRIDGE_CM_DIR   = DIR / "data" / "Bridge" / "cm"
BRIDGE_CL_DIR   = DIR / "data" / "Bridge" / "cl"
BRIDGE_VEL_DIR  = DIR / "data" / "Bridge" / "vel"
BRIDGE_FY_DIR   = DIR / "data" / "Bridge" / "force"

# 19.5 m/s is left out of the bridge dataset (outlier case, decided with the supervisors).
BRIDGE_EXCLUDED_RAW_SPEEDS = {"19.5"}

BASE_DTYPES  = {"step": "int32", "time": "float32"}
VALUE_DTYPE  = "float32"
# Steps dropped at the start of every case to remove the impulsive start.
# Override with the environment variable VIV_INITIAL_TRIM.
INITIAL_TRIM_STEPS = int(os.getenv("VIV_INITIAL_TRIM", "100"))


def read_out_files(filepath: str | Path) -> tuple[pd.DataFrame, str]:
    """Parse one Fluent .out monitor file into columns step, val, time (header lines are skipped)."""
    try:
        data    = []
        started = False
        with open(filepath, "r") as f:
            for line in f:
                s = line.strip()
                if not started:
                    if s and (s[0].isdigit()
                              or (s[0] == "-" and len(s) > 1 and s[1].isdigit())):
                        started = True
                    else:
                        continue
                parts = s.split()
                if len(parts) >= 3:
                    try:
                        data.append([int(float(parts[0])),
                                     float(parts[1]),
                                     float(parts[2])])
                    except ValueError:
                        continue

        fname = os.path.basename(str(filepath))
        df    = pd.DataFrame(data, columns=["step", "val", "time"])
        if df.empty:
            return df.astype({**BASE_DTYPES, "val": VALUE_DTYPE}), fname
        df = (df.astype({**BASE_DTYPES, "val": VALUE_DTYPE})
                .sort_values(["time", "step"]))
        return df, fname

    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return (pd.DataFrame(columns=["step", "val", "time"])
                  .astype({**BASE_DTYPES, "val": VALUE_DTYPE}),
                os.path.basename(str(filepath)))


def extract_case_name(filepath: str | Path) -> str:
    """Case name from a file name: 'UrX' labels are normalised, otherwise the part after the last '-' or '_'."""
    stem = Path(filepath).stem

    ur_match = re.search(r"[Uu][Rr][_\-]?([0-9]+(?:\.[0-9]+)?)", stem)
    if ur_match:
        return format_ur_label(float(ur_match.group(1)))

    if "-" in stem:
        return stem.rsplit("-", 1)[1]

    if "_" in stem:
        return stem.rsplit("_", 1)[1]

    return stem


def _resolve_data_dirs(dataset: str) -> tuple[Path, Path, Path]:
    """Directories of the disp, drag/moment and lift files for a dataset."""
    ds = dataset.strip().lower()
    if ds == "bridge":
        return BRIDGE_DISP_DIR, BRIDGE_CM_DIR, BRIDGE_CL_DIR
    if ds in CYLINDER200_ALIASES:
        return CYLINDER200_DISP_DIR, CYLINDER200_CD_DIR, CYLINDER200_CL_DIR
    raise ValueError(
        f"Unknown dataset: '{ds}'. Must be one of:cylinder200, bridge."
    )


def _normalize_bridge_cases_to_ur(df: pd.DataFrame, fn_hz: float, d_ref: float) -> pd.DataFrame:
    """Replace bridge wind-speed labels by Ur labels: Ur = U / (fn D)."""
    if fn_hz <= 0 or d_ref <= 0:
        raise ValueError("fn_hz and d_ref must be positive.")
    out   = df.copy()
    speed = pd.to_numeric(out["case"].astype(str), errors="coerce")
    bad   = int(speed.isna().sum())
    if bad:
        raise ValueError(
            f"{bad} bridge case labels could not be parsed as velocities. "
            "Expected filenames like 'disp-16.11.out'."
        )
    out["case"] = (speed / float(fn_hz * d_ref)).map(format_ur_label).astype("string")
    return out


def empty_case_frame(value_name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=["case", "step", "time", value_name]).astype(
        {"case": "string", **BASE_DTYPES, value_name: VALUE_DTYPE}
    )


def read_out_directory(directory: Path, value_name: str) -> pd.DataFrame:
    """Read all .out files of one signal into a long dataframe (case, step, time, value)."""
    files = sorted(directory.glob("*.out"))
    if not files:
        print(f"No .out files found in: {directory}")
        return empty_case_frame(value_name)

    frames = []
    for f in files:
        df, _ = read_out_files(str(f))
        if not df.empty:
            case_df = df[["step", "time", "val"]].copy()
            case_df.insert(0, "case", extract_case_name(f))
            if INITIAL_TRIM_STEPS > 0:
                case_df = case_df.iloc[INITIAL_TRIM_STEPS:].reset_index(drop=True)
            if case_df.empty:
                continue
            frames.append(case_df)

    if not frames:
        return empty_case_frame(value_name)

    combined = (
        pd.concat(frames, ignore_index=True)
          .sort_values(["case", "time", "step"])
          .drop_duplicates(subset=["case", "step", "time"], keep="last")
          .reset_index(drop=True)
    )
    return (combined.rename(columns={"val": value_name})
                    .astype({"case": "string", **BASE_DTYPES, value_name: VALUE_DTYPE}))


def downsample(df: pd.DataFrame, every_n: int) -> pd.DataFrame:
    """Keep every n-th row of each case."""
    if every_n <= 1:
        return df.copy()
    pos = df.groupby("case", sort=False).cumcount()
    return df[pos % every_n == 0].reset_index(drop=True)


def merge_dataframes(
    dataset: str   | None = None,
    fn_hz:   float | None = None,
    d_ref:   float | None = None,
    convert_bridge_to_ur: bool = True,
) -> pd.DataFrame:
    """Load and join displacement, drag/moment and lift for all cases of a dataset.

    Only time steps present in all three signals are kept. For the bridge, the
    excluded speeds are removed and, if convert_bridge_to_ur is True, cases are
    relabelled by Ur (needs fn_hz and d_ref).
    """
    ds = (dataset or os.getenv("VIV_DATASET", "cylinder200")).strip().lower()
    disp_dir, cd_dir, cl_dir = _resolve_data_dirs(ds)

    disp_df = read_out_directory(disp_dir, "disp")
    cd_df   = read_out_directory(cd_dir,   "cd")
    cl_df   = read_out_directory(cl_dir,   "cl")

    case_sets = {name: set(d["case"].unique()) for name, d in
                 (("disp", disp_df), ("cd", cd_df), ("cl", cl_df)) if not d.empty}
    if len(case_sets) > 1:
        all_cases = set.union(*case_sets.values())
        for name, cases in case_sets.items():
            missing = all_cases - cases
            if missing:
                print(f"WARNING: dataset='{ds}' signal='{name}' is missing "
                      f"case(s) {sorted(missing)} present in other signals "
                      f"-- those cases will be dropped by the inner join below.")

    empty = pd.DataFrame(
        columns=["case", "step", "time", "disp", "cd", "cl"]
    ).astype({"case": "string", **BASE_DTYPES,
              "disp": VALUE_DTYPE, "cd": VALUE_DTYPE, "cl": VALUE_DTYPE})

    if disp_df.empty or cd_df.empty or cl_df.empty:
        print(f"WARNING: one or more signal directories empty for dataset='{ds}'")
        return empty

    df = (disp_df
          .merge(cd_df,  on=["case", "step", "time"], how="inner", validate="one_to_one")
          .merge(cl_df,  on=["case", "step", "time"], how="inner", validate="one_to_one")
          .sort_values(["case", "time", "step"])
          .reset_index(drop=True))

    if ds == "bridge" and BRIDGE_EXCLUDED_RAW_SPEEDS:
        excluded_mask = df["case"].astype(str).isin(BRIDGE_EXCLUDED_RAW_SPEEDS)
        if excluded_mask.any():
            print(f"Excluding bridge case(s) {sorted(df.loc[excluded_mask, 'case'].unique())} "
                  f"m/s per BRIDGE_EXCLUDED_RAW_SPEEDS ({excluded_mask.sum()} rows dropped).")
            df = df[~excluded_mask].reset_index(drop=True)

    if ds == "bridge" and convert_bridge_to_ur:
        if fn_hz is None:
            env = os.getenv("BRIDGE_FN_HZ")
            fn_hz = float(env) if env else None
        if d_ref is None:
            env = os.getenv("BRIDGE_D_REF")
            d_ref = float(env) if env else None
        if fn_hz is None or d_ref is None:
            raise ValueError(
                "Bridge dataset selected but fn_hz/d_ref are missing.\n"
                "Call merge_dataframes(dataset='bridge', fn_hz=0.32, d_ref=7.42)."
            )
        df = _normalize_bridge_cases_to_ur(df, fn_hz=fn_hz, d_ref=d_ref)

    return df


def _load_force(dataset: str) -> dict[str, pd.DataFrame]:
    """Read the optional velocity and fluid-force monitors (used to compute acceleration)."""
    ds = dataset.strip().lower()
    known = {"bridge"} | CYLINDER200_ALIASES
    if ds not in known:
        return {}

    out = {}
    if ds in CYLINDER200_ALIASES:
        if CYLINDER200_VEL_DIR.exists():
            v = read_out_directory(CYLINDER200_VEL_DIR, "vel")
            if not v.empty: out["vel"] = v
        if CYLINDER200_FY_DIR.exists():
            fdf = read_out_directory(CYLINDER200_FY_DIR, "force")
            if not fdf.empty: out["force"] = fdf
        return out

    fn_hz = float(config["bridge_fn_hz"]); d_ref = float(config["bridge_D_ref"])
    if BRIDGE_VEL_DIR.exists():
        v = read_out_directory(BRIDGE_VEL_DIR, "vel")
        if not v.empty:
            out["vel"] = _normalize_bridge_cases_to_ur(v, fn_hz=fn_hz, d_ref=d_ref)
    if BRIDGE_FY_DIR.exists():
        fdf = read_out_directory(BRIDGE_FY_DIR, "force")
        if not fdf.empty:
            out["force"] = _normalize_bridge_cases_to_ur(fdf, fn_hz=fn_hz, d_ref=d_ref)
    return out


def compute_kinematics(df: pd.DataFrame, dataset: str | None = None,
                       structural_params: dict | None = None,
                       bridge_structural_params: dict | None = None,
                       acc_source: str = "force_residual") -> pd.DataFrame:
    """Add the columns vel and acc to the merged dataframe.

    vel is the recorded velocity monitor where available. acc depends on acc_source:
      "force_residual": acc = (F_fluid - c v - k y) / m with the same m, c, k as the
                        Newmark integrator. This makes C_L an exact linear function of
                        [disp, vel, acc] (the acceleration leakage of thesis Sec. 5.4).
      "savgol_vel":     numerical derivative of the smoothed recorded velocity.
    Cases without monitors fall back to Savitzky-Golay derivatives of the displacement.
    """
    if acc_source not in ("force_residual", "savgol_vel"):
        raise ValueError(
            f"acc_source must be 'force_residual' or 'savgol_vel', got {acc_source!r}")

    print(f"Computing velocity and acceleration... (acc_source={acc_source})")
    df = df.sort_values(["case", "time", "step"]).reset_index(drop=True)

    force_signal = _load_force(dataset) if dataset else {}
    ds = (dataset or "").strip().lower()

    sp = None
    if ds == "bridge":
        sp = bridge_structural_params if bridge_structural_params is not None else None
    elif structural_params is not None:
        sp = structural_params

    if force_signal and sp is not None:
        m = sp['m']
        c = sp['c']
        k = sp['k']
        print(f"[compute_kinematics] m={sp['m']:.6e}  "
              f"c={sp['c']:.6e}  k={sp['k']:.6e}")

        if "vel" in force_signal:
            df = df.merge(force_signal["vel"], on=["case", "step", "time"], how="left", validate="one_to_one")
        if "force" in force_signal:
            df = df.merge(force_signal["force"], on=["case", "step", "time"], how="left", validate="one_to_one")
            if acc_source == "force_residual":
                F_fluid = df["force"].astype("float32")
                y       = df["disp"].astype("float32")
                v       = df["vel"].astype("float32")
                df["acc"] = (F_fluid - c * v - k * y) / float(m)
            df = df.drop(columns=["force"])

        if acc_source == "savgol_vel" and "vel" in df.columns and not df["vel"].isna().all():
            df["acc"] = _savgol_derivative_per_case(df, "vel")

        needs_fill = (
            "vel" not in df.columns or df["vel"].isna().any() or
            "acc" not in df.columns or df["acc"].isna().any()
        )

        if needs_fill:
            print("  WARNING: some cases missing force/vel data, using Savgol fallback.")
            df = _fill_missing_kinematics_with_savgol(df)

        return df

    return _fill_missing_kinematics_with_savgol(df)


def _savgol_derivative_per_case(df: pd.DataFrame, col: str) -> pd.Series:
    """Time derivative of one column per case, after Savitzky-Golay smoothing (window 11, order 3)."""
    out = pd.Series(index=df.index, dtype="float64")
    for _, case_df in df.groupby("case", sort=False):
        t = case_df["time"].to_numpy()
        x = case_df[col].to_numpy(dtype="float64")
        if len(x) < 11:
            d = np.gradient(x, t)
        else:
            x_smooth = savgol_filter(x, window_length=11, polyorder=3)
            d = np.gradient(x_smooth, t)
        out.loc[case_df.index] = d
    return out


def _fill_missing_kinematics_with_savgol(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing vel/acc from Savitzky-Golay derivatives of the displacement."""
    velocities, accelerations = [], []
    for _, case_df in df.groupby("case", sort=False):
        t = case_df["time"].to_numpy()
        d = case_df["disp"].to_numpy()
        if len(d) < 11:
            v = np.gradient(d, t)
            a = np.gradient(v, t)
        else:
            d_smooth = savgol_filter(d, window_length=11, polyorder=3)
            v = np.gradient(d_smooth, t)
            a = np.gradient(v, t)
        velocities.extend(v)
        accelerations.extend(a)

    if "vel" in df.columns:
        df["vel"] = df["vel"].fillna(pd.Series(velocities, index=df.index))
    else:
        df["vel"] = velocities
    if "acc" in df.columns:
        df["acc"] = df["acc"].fillna(pd.Series(accelerations, index=df.index))
    else:
        df["acc"] = accelerations
    return df


# Increase this when the preprocessing changes, so an old cache is not reused.
BRIDGE_CACHE_VERSION = 2


def bridge_cache_path() -> Path:
    """Path of the parquet cache; the name encodes downsampling, trim and cache version."""
    return (
        PROJECT_ROOT / "data" / "cache"
        / f"bridge_ds{config['bridge_downsample']}"
        f"_trim{INITIAL_TRIM_STEPS}"
        f"_v{BRIDGE_CACHE_VERSION}.parquet"
    )


def load_bridge_df_cached(
    fn_hz: float,
    d_ref: float,
    bridge_structural_params: dict,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Return the preprocessed bridge dataframe, building and caching it on first use.

    Pipeline: merge_dataframes -> downsample (every 20th sample) -> compute_kinematics.
    Use force_rebuild=True (or delete the cache file) after changing the raw data.
    """
    cache = bridge_cache_path()

    if cache.exists() and not force_rebuild:
        print(f"[cache] loading preprocessed bridge df from {cache}")
        df = pd.read_parquet(cache)
        expected = {"case", "step", "time", "disp", "cd", "cl", "vel", "acc"}
        missing = expected - set(df.columns)
        if missing or df.empty:
            raise RuntimeError(
                f"Bridge cache at {cache} is invalid "
                f"(empty={df.empty}, missing={missing}). "
                f"Delete it and rebuild."
            )
        return df

    print("[cache] no cache found — building (this is the slow path, ~1h)")
    raw = merge_dataframes(dataset="bridge", fn_hz=fn_hz, d_ref=d_ref)
    if raw.empty:
        raise RuntimeError("merge_dataframes returned empty — check data dirs.")
    raw = downsample(raw, config["bridge_downsample"])
    raw = compute_kinematics(
        raw,
        dataset="bridge",
        bridge_structural_params=bridge_structural_params,
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + f".tmp.{os.getpid()}")
    raw.to_parquet(tmp, index=False)
    tmp.rename(cache)
    print(f"[cache] wrote {cache}  ({cache.stat().st_size/1e6:.0f} MB)")
    return raw
