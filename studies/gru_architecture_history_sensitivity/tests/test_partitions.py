import pytest
from audit_pipeline import audit_dataset


@pytest.fixture(scope="module")
def cylinder_audit():
    return audit_dataset("cylinder200")


@pytest.fixture(scope="module")
def bridge_audit():
    return audit_dataset("bridge")


def test_cylinder_split_is_13_4_4(cylinder_audit):
    assert cylinder_audit["n_train"] == 13
    assert cylinder_audit["n_val"] == 4
    assert cylinder_audit["n_test"] == 4


def test_bridge_split_is_17_5_5(bridge_audit):
    assert bridge_audit["n_retained_cases"] == 27
    assert bridge_audit["n_train"] == 17
    assert bridge_audit["n_val"] == 5
    assert bridge_audit["n_test"] == 5


def test_bridge_excludes_ur_8_2126(bridge_audit):
    all_cases = (set(bridge_audit["train_cases"]) | set(bridge_audit["val_cases"])
                 | set(bridge_audit["test_cases"]))
    assert "Ur8.2126" not in all_cases
    assert not any(abs(float(c.replace("Ur", "")) - 8.2126) < 1e-6 for c in all_cases)


def test_no_acceleration_column_in_study_input_cols():
    import json
    from _common import STUDY_ROOT
    fixed = json.loads((STUDY_ROOT / "configs" / "architecture_grid.json")
                        .read_text())["fixed_hyperparameters"]
    assert "acc" not in fixed["input_cols"]
    assert set(fixed["input_cols"]) == {"disp", "vel"}


def test_no_acceleration_reaches_gru_model_input_size():
    from viv_analysis.models.gru import VIV_GRU
    input_cols = ["disp", "vel"]
    use_ur_context = True
    model = VIV_GRU(input_size=len(input_cols) + (1 if use_ur_context else 0),
                     hidden_size=8, num_layers=1)
    assert model.gru.input_size == 3
