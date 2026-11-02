"""When automatic retraining is due: by the labelled items + connections added since the last
one, not the number of sheets."""
import json
import os

from pid_dexpi_aas.image.training_export import graphml_bytes, retrain_status


def _sheet(root, subset, name, n_items, n_links):
    syms = [(f"s{i}", "valve", (i * 10.0, 0.0, i * 10.0 + 5, 5.0)) for i in range(n_items)]
    edges = [(f"s{i}", f"s{i + 1}", "solid") for i in range(n_links)]
    d = root / "data" / "PID2Graph" / "Complete" / subset
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.graphml").write_bytes(graphml_bytes(syms, edges))


def test_retraining_is_due_by_the_value_of_new_data(tmp_path):
    data = tmp_path / "data" / "PID2Graph"
    (tmp_path / "ml" / "models").mkdir(parents=True)
    _sheet(tmp_path, "Team v3", "lumen", 600, 400)          # 1000 hand-labelled units: due at 10 % = 100 -> 150
    _sheet(tmp_path, "Reviewed", "small", 30, 20)            # 50 units: not yet
    st = retrain_status(data)
    assert (st["new_units"], st["threshold"], st["due"]) == (50, 150, False)
    _sheet(tmp_path, "Reviewed", "big", 70, 50)              # +120: one big P&ID counts more than a sketch
    st = retrain_status(data)
    assert st["new_units"] == 170 and st["due"]
    # tried on this data (whatever came of it): not again until more is added
    (tmp_path / "ml" / "models" / "retrain_state.json").write_text(json.dumps({"tried_units": 170}))
    assert not retrain_status(data)["due"]
    # trained into the live model: no longer new; a sheet replaced since is new again
    trained = {n: v for n, v in st["sheets"].items()}
    (tmp_path / "ml" / "models" / "retrain_state.json").write_text(json.dumps({"trained": trained}))
    assert retrain_status(data)["new_units"] == 0
    g = data / "Complete" / "Reviewed" / "big.graphml"
    os.utime(g, (g.stat().st_atime, g.stat().st_mtime + 10))
    assert retrain_status(data)["new"] == [{"name": "big", "units": 120}]
