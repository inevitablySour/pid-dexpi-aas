"""Automatic retraining when enough new reviewed sheets have been added (run nightly by cron).

    .venv/bin/python ml/pid2graph/auto_retrain.py --check        # what is new, is it due
    .venv/bin/python ml/pid2graph/auto_retrain.py --if-due       # cron: retrain only when due
    .venv/bin/python ml/pid2graph/auto_retrain.py --force        # retrain now
    .venv/bin/python ml/pid2graph/auto_retrain.py --approve      # deploy the candidate held back

Due when the reviewed sheets added since the last retraining hold enough labelled items +
connections (image/training_export.retrain_status: 10 % of the hand-labelled data, at least
150). Then:
  1. a candidate model is trained like the live one (pid2graph_final.yaml - v3) with the
     reviewed sheets ("Reviewed", made in the app: "Add this to the training data") added to
     its training data - ~4 h on the GPU;
  2. the candidate and the live model are scored on the held-out sheets ("Reviewed test" -
     never trained on); the candidate replaces the live model only if it is no worse on items
     and on connections (within TOLERANCE). The old model is kept in ml/models/archive/;
  3. without held-out sheets nothing can be compared: the candidate waits for --approve.
Everything is recorded in ml/models/retrain_state.json (the app shows it) and ml/logs/auto_retrain.log.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pid_dexpi_aas.image.training_export import retrain_state_path, retrain_status  # noqa: E402

DATA = ROOT / "data" / "PID2Graph"
MODELS = ROOT / "ml" / "models"
LIVE = MODELS / "pid2graph_final.pt"
BASE = ROOT / "ml" / "relationformer" / "trained_weights" / "runs" / "sheetsplit_bs4_10" / "models" / "checkpoint_epoch=150.pt"
ML_PY = ROOT / "ml" / "relationformer" / ".venv" / "bin" / "python"
TOLERANCE = 0.02  # a candidate may score this much lower and still replace the live model: two identical
                  # trainings differ by ~0.02 (v3 fold 0 trained twice, 2026-10-15: 0.704/0.415 vs 0.688/0.394)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M')}] {msg}", flush=True)


def state() -> dict:
    p = retrain_state_path(DATA)
    return json.loads(p.read_text()) if p.exists() else {}


def save_state(s: dict) -> None:
    retrain_state_path(DATA).write_text(json.dumps(s, indent=1))


def training_running() -> bool:
    out = subprocess.run(["pgrep", "-f", "train_pid2graph.py"], capture_output=True, text=True).stdout
    return bool(out.strip())


def write_config() -> str:
    """pid2graph_final.yaml (the live model's) with the reviewed sheets as training data."""
    import yaml
    cfg = yaml.safe_load(open(ROOT / "ml" / "pid2graph" / "pid2graph_final.yaml"))
    extra = list(cfg["DATA"].get("EXTRA_TRAIN_SUBSETS") or [])
    if "Patched/Reviewed" not in extra:
        extra.append("Patched/Reviewed")
    cfg["DATA"]["EXTRA_TRAIN_SUBSETS"] = extra
    name = "pid2graph_final_auto.yaml"
    with open(ROOT / "ml" / "pid2graph" / name, "w") as f:
        f.write("# written by auto_retrain.py: pid2graph_final.yaml + the reviewed sheets (Patched/Reviewed)\n")
        yaml.safe_dump(cfg, f, sort_keys=False)
    return name


def score(checkpoint: Path) -> dict | None:
    """Items / connections mAP on the held-out reviewed sheets (eval_sheets.py --set reviewed)."""
    out = MODELS / "candidates" / f"score_{checkpoint.stem}.json"
    r = subprocess.run([str(ML_PY), str(ROOT / "ml" / "pid2graph" / "eval" / "eval_sheets.py"), "--checkpoint",
                        str(checkpoint), "--set", "reviewed", "--out", str(out)],
                       cwd=ROOT / "ml" / "relationformer", capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        log(f"scoring {checkpoint.name} failed: {r.stderr[-800:]}")
        return None
    d = json.loads(out.read_text())
    return {"items": round(d["node_mAP"], 4), "links": round(d["edge_mAP"], 4)}


def deploy(candidate: Path, s: dict, record: dict) -> None:
    (MODELS / "archive").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    if LIVE.exists():
        shutil.copy2(LIVE, MODELS / "archive" / f"pid2graph_final_{stamp}.pt")
    shutil.copy2(candidate, LIVE)  # the app picks it up: its result cache is keyed by the model file's time
    s["trained"] = record["sheets"]
    s.pop("pending", None)
    record["deployed"] = True
    log(f"deployed {candidate.name} as the live model (old one archived as pid2graph_final_{stamp}.pt)")


def run(force: bool) -> None:
    st = retrain_status(DATA)
    log(f"new reviewed data: {st['new_units']} units in {len(st['new'])} sheets (due at {st['threshold']}); "
        f"held-out sheets: {st['held_out']}")
    if not force and not st["due"]:
        log("not due")
        return
    if training_running():
        log("a training is running - trying again next time")
        return
    s = state()
    s["tried_units"] = st["new_units"]  # never the same data twice in a row, whatever the outcome
    save_state(s)
    (MODELS / "candidates").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = f"candidates/pid2graph_auto_{stamp}.pt"
    log(f"training {out} (~4 h) ...")
    env = dict(os.environ, EXP=f"auto_{stamp}", CONFIG=write_config(), OUT=out)
    r = subprocess.run(["bash", str(ROOT / "ml" / "pid2graph" / "run_final.sh"), str(BASE)], env=env,
                       stdout=open(ROOT / "ml" / "logs" / f"auto_retrain_{stamp}.log", "w"), stderr=subprocess.STDOUT)
    candidate = MODELS / out
    record = {"when": stamp, "candidate": out, "new_units": st["new_units"],
              "new_sheets": [x["name"] for x in st["new"]], "sheets": st["sheets"], "deployed": False}
    if r.returncode != 0 or not candidate.exists():
        record["result"] = "training failed - see ml/logs/auto_retrain_" + stamp + ".log"
        log(record["result"])
    elif not st["held_out"]:
        record["result"] = ("trained, not deployed: no held-out sheets to compare on - approve it in the app "
                            "(or --approve), or add held-out sheets")
        s["pending"] = record
        log(record["result"])
    else:
        cand, live = score(candidate), score(LIVE)
        record.update(candidate_score=cand, live_score=live)
        if cand and live and cand["items"] >= live["items"] - TOLERANCE and cand["links"] >= live["links"] - TOLERANCE:
            record["result"] = f"deployed: held-out items {live['items']} -> {cand['items']}, links {live['links']} -> {cand['links']}"
            deploy(candidate, s, record)
        else:
            record["result"] = f"kept the live model: candidate scored {cand} vs live {live} on the held-out sheets"
            log(record["result"])
    s.setdefault("history", []).append(record)
    save_state(s)


def approve() -> None:
    s = state()
    pending = s.get("pending")
    if not pending:
        log("nothing waiting for approval")
        return
    deploy(MODELS / pending["candidate"], s, pending)
    pending["result"] += " - approved by hand"
    s.setdefault("history", []).append(dict(pending, approved=True))
    save_state(s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--if-due", action="store_true")
    g.add_argument("--force", action="store_true")
    g.add_argument("--approve", action="store_true")
    a = ap.parse_args()
    if a.check:
        st = retrain_status(DATA)
        print(json.dumps({k: v for k, v in st.items() if k != "sheets"}, indent=1))
    elif a.approve:
        approve()
    else:
        run(force=a.force)


if __name__ == "__main__":
    main()
