import pandas as pd

from select_configuration import build_pareto_table

_CYLINDER_BASE = {
    "dataset": "cylinder200", "stage": 1, "seq_len": 1000, "history_label": None,
    "n_seeds": 3, "seeds": [123, 456, 789], "all_valid": True,
    "open_loop_val_r2_median": 0.99, "open_loop_val_r2_iqr": 0.0,
    "trainable_parameter_count_median": 10000.0,
    "closed_loop_mean_inference_time_s_median": 5.0,
}


def _cyl_row(H, L, stable, median_amp, worst_amp, freq):
    return {
        **_CYLINDER_BASE, "hidden_size": H, "num_layers": L,
        "closed_loop_validation_stable_count_median": stable,
        "closed_loop_validation_median_abs_rel_amp_error_median": median_amp,
        "closed_loop_validation_worst_abs_rel_amp_error_median": worst_amp,
        "closed_loop_validation_median_abs_f_osc_rel_error_median": freq,
    }


def test_hierarchy_orders_by_stable_count_then_median_then_worst_then_freq():
    agg = pd.DataFrame([
        _cyl_row(64, 2, stable=4, median_amp=0.02, worst_amp=0.05, freq=0.01),
        _cyl_row(32, 1, stable=2, median_amp=0.001, worst_amp=0.001, freq=0.001),
        _cyl_row(32, 2, stable=4, median_amp=0.05, worst_amp=0.05, freq=0.01),
        _cyl_row(128, 1, stable=4, median_amp=0.02, worst_amp=0.10, freq=0.01),
        _cyl_row(128, 2, stable=4, median_amp=0.02, worst_amp=0.05, freq=0.05),
    ])
    ranked = build_pareto_table("cylinder200", agg)
    order = list(zip(ranked["hidden_size"].tolist(), ranked["num_layers"].tolist()))
    assert order == [(64, 2), (128, 2), (128, 1), (32, 2), (32, 1)], (
        "expected A > E > D > C > B: B must rank LAST despite having the "
        f"single lowest amplitude/frequency error of any row, got {order}")


def test_stable_count_alone_beats_a_much_lower_amplitude_error():
    agg = pd.DataFrame([
        _cyl_row(64, 2, stable=4, median_amp=0.15, worst_amp=0.15, freq=0.15),
        _cyl_row(32, 1, stable=1, median_amp=0.001, worst_amp=0.001, freq=0.001),
    ])
    ranked = build_pareto_table("cylinder200", agg)
    top = ranked.iloc[0]
    assert (int(top["hidden_size"]), int(top["num_layers"])) == (64, 2)


_BRIDGE_BASE = {
    "dataset": "bridge", "stage": 1, "seq_len": 2500, "history_label": None,
    "n_seeds": 3, "seeds": [123, 456, 789], "all_valid": True,
    "open_loop_val_r2_median": 0.99, "open_loop_val_r2_iqr": 0.0,
    "trainable_parameter_count_median": 10000.0,
    "closed_loop_mean_inference_time_s_median": 5.0,
}


def _bridge_row(H, L, stable, median_amp, worst_amp, freq):
    return {
        **_BRIDGE_BASE, "hidden_size": H, "num_layers": L,
        "closed_loop_validation_stable_count_among_settled_lco_median": stable,
        "closed_loop_validation_median_abs_rel_amp_error_among_settled_lco_median": median_amp,
        "closed_loop_validation_worst_abs_rel_amp_error_among_settled_lco_median": worst_amp,
        "closed_loop_validation_median_abs_f_osc_rel_error_median": freq,
    }


def test_bridge_ranking_reads_the_settled_lco_column_variants():
    agg = pd.DataFrame([
        _bridge_row(32, 1, stable=2, median_amp=0.01, worst_amp=0.01, freq=0.01),
        _bridge_row(64, 2, stable=5, median_amp=0.10, worst_amp=0.20, freq=0.05),
    ])
    ranked = build_pareto_table("bridge", agg)
    order = list(zip(ranked["hidden_size"].tolist(), ranked["num_layers"].tolist()))
    assert order == [(64, 2), (32, 1)], (
        f"bridge ranking must prioritize settled-LCO stable count over "
        f"amplitude/frequency error, same as cylinder200, got {order}")
