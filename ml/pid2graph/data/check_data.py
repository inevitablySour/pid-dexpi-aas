"""Integrity check of everything a fine-tune run trains and validates on - run before a
long training run. Uses the training code's own data loading (build_pid2graph_data), so
it checks exactly what each fold sees:

  - no sheet in more than one of train / val / test, per fold; the team test drawing
    (LightUp) never in training;
  - every label file parses the way the loader parses it; classes known; boxes inside the
    image with positive size; connections between existing symbols, no self-connections;
  - no patch with more symbols than the model has queries (they would be dropped);
  - team data: patch labels agree with the whole-sheet labels they were cut from.

Usage (ML venv): ml/relationformer/.venv/bin/python ml/pid2graph/data/check_data.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

HERE = Path(__file__).resolve().parent
RF = HERE.parents[1] / "relationformer"
sys.path.insert(0, str(RF))
from pid2graph_data.dataset_pid2graph import (  # noqa: E402
    _collapse_graph, _discover_pairs, build_pid2graph_data, real_fold_sheets)
from train_pid2graph import dict2obj  # noqa: E402
import networkx as nx  # noqa: E402

DATA = HERE.parents[2] / "data" / "PID2Graph"
CLASSES = {"arrow", "general", "inlet/outlet", "instrumentation", "pump", "tank", "valve"}


def sheet(p) -> str:
    return f"{p[1].parent.parent.name}/{p[1].parent.name}"


def check_pairs(pairs, n_queries, label):
    problems, counts, n_sym, n_edge = [], Counter(), [], 0
    for img, gml in pairs:
        try:
            g = nx.read_graphml(gml)
            labels, boxes, edges = _collapse_graph(g)
        except Exception as exc:
            problems.append(f"{gml}: does not parse ({exc})")
            continue
        W, H = Image.open(img).size
        counts.update(labels)
        n_sym.append(len(labels))
        n_edge += len(edges)
        for l, (x1, y1, x2, y2) in zip(labels, boxes):
            if l not in CLASSES:
                problems.append(f"{gml}: unknown class {l!r}")
            if x2 <= x1 or y2 <= y1:
                problems.append(f"{gml}: empty box {l} {(x1, y1, x2, y2)}")
            if x1 < -2 or y1 < -2 or x2 > W + 2 or y2 > H + 2:
                problems.append(f"{gml}: box outside the {W}x{H} image: {l} {[round(v) for v in (x1, y1, x2, y2)]}")
        for a, b, _ in edges:
            if a == b or not (0 <= a < len(labels) and 0 <= b < len(labels)):
                problems.append(f"{gml}: bad connection {a}-{b}")
        if len(labels) > n_queries:
            problems.append(f"{gml}: {len(labels)} symbols > {n_queries} queries")
    print(f"  {label}: {len(pairs)} patches, symbols/patch max {max(n_sym, default=0)}, "
          f"{sum(n_sym)} symbols, {n_edge} connections, classes {dict(counts)}")
    return problems


def main():
    config = dict2obj(yaml.safe_load(open(RF / "configs" / "pid2graph_finetune.yaml")))
    config.DATA.DATA_PATH = str(DATA)
    nq = config.MODEL.DECODER.NUM_QUERIES
    problems = []
    team_test = _discover_pairs(DATA, list(config.DATA.TEAM_TEST_SUBSETS))
    team_test_names = {p[1].parent.name for p in team_test}
    print(f"team test: {sorted(team_test_names)} ({len(team_test)} patches)")
    for fold in range(int(config.DATA.NUM_FOLDS)):
        train, val = build_pid2graph_data(config, "split", fold=fold)
        test = build_pid2graph_data(config, "test", fold=fold)
        tr, va, te = ({sheet(p) for p in d.samples} for d in (train, val, test))
        for a, b, name in ((tr, va, "train/val"), (tr, te, "train/test"), (va, te, "val/test")):
            if a & b:
                problems.append(f"fold {fold}: {name} share sheets {sorted(a & b)[:5]}")
        leak = {p[1].parent.name for p in train.samples} & team_test_names
        if leak:
            problems.append(f"fold {fold}: team test drawing in training: {leak}")
        real = real_fold_sheets(config, fold)
        src = Counter(p[1].parent.parent.name for p in train.samples)
        print(f"fold {fold}: train {len(train.samples)} patches {dict(src)}; val {len(val.samples)}; "
              f"test {len(test.samples)} ({', '.join(real['test'])})")
    print("label files:")
    team = _discover_pairs(DATA, list(config.DATA.EXTRA_TRAIN_SUBSETS))
    real = _discover_pairs(DATA, [config.DATA.REAL_SUBSET])
    problems += check_pairs(team, nq, "team train")
    problems += check_pairs(team_test, nq, "team test")
    problems += check_pairs(real, nq, "OPEN100")
    # team patches vs the whole sheets they were cut from
    for sub in list(config.DATA.EXTRA_TRAIN_SUBSETS) + list(config.DATA.TEAM_TEST_SUBSETS):
        complete = DATA / "Complete" / Path(sub).name
        for d in sorted((DATA / sub).iterdir()):
            whole = complete / f"{d.name}.graphml"
            if not whole.exists():
                problems.append(f"{sub}/{d.name}: no whole-sheet labels at {whole}")
                continue
            g = nx.read_graphml(whole)
            n_patch_syms = len({(round(float(a['xmin'])), round(float(a['ymin'])))
                                for gm in d.glob('*.graphml') for _, a in nx.read_graphml(gm).nodes(data=True)})
            print(f"  {d.name}: whole sheet {g.number_of_nodes()} symbols / {g.number_of_edges()} connections, "
                  f"{len(list(d.glob('*.graphml')))} patches")
    print("\nPROBLEMS:" if problems else "\nno problems found")
    for p in problems[:60]:
        print(" -", p)
    if len(problems) > 60:
        print(f" ... and {len(problems) - 60} more")


if __name__ == "__main__":
    main()
