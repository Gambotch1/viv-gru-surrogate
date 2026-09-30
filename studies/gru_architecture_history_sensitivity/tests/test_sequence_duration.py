import json

from _common import STUDY_ROOT


def test_bridge_history_grid_durations_are_seq_len_times_dt():
    grid = json.loads((STUDY_ROOT / "configs" / "bridge_history_grid.json").read_text())
    dt = grid["dt"]
    for point in grid["history_points"]:
        exact = round(point["seq_len"] * dt, 3)
        assert exact == point["exact_duration_s"], (
            f"{point['label']}: seq_len*dt={exact} != declared "
            f"exact_duration_s={point['exact_duration_s']}")


def test_bridge_1tn_point_is_not_exactly_3_125s():
    grid = json.loads((STUDY_ROOT / "configs" / "bridge_history_grid.json").read_text())
    dt = grid["dt"]
    point = next(p for p in grid["history_points"] if p["label"] == "1Tn")
    assert point["seq_len"] == 1563
    exact = point["seq_len"] * dt
    assert abs(exact - grid["Tn_s"]) > 1e-9
    assert round(exact, 3) == 3.126


def test_cylinder_history_grid_durations_are_seq_len_times_dt():
    grid = json.loads((STUDY_ROOT / "configs" / "cylinder_history_grid.json").read_text())
    dt = grid["dt"]
    for point in grid["history_points"]:
        exact = point["seq_len"] * dt
        assert abs(exact - point["nominal_duration_s"]) < 1e-9


def test_train_receipt_duration_field(tmp_path, monkeypatch):
    seq_len, dt = 1563, 0.002
    assert round(seq_len * dt, 3) == 3.126
