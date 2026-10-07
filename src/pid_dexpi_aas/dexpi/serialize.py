"""Serialize a built DexpiModel to Proteus/DEXPI XML and verify the round-trip.

Uses PatchedProteusSerializer (see proteus_patch.py), not the stock ProteusSerializer:
pyDEXPI 1.2.0's exporter drops piping/instrumentation data silently, so the stock
serializer would pass this round-trip check while losing real model content. See
../../README.md for the full writeup.
"""
from __future__ import annotations

from pathlib import Path

from pydexpi.dexpi_classes.dexpiModel import DexpiModel

from pid_dexpi_aas.dexpi.proteus_patch import PatchedProteusSerializer


def save_and_verify(model: DexpiModel, dir_path: str | Path, filename: str) -> DexpiModel:
    """Write `model` to `<dir_path>/<filename>` and re-load it, asserting the list-field
    counts on the conceptual model survive the round-trip unchanged. Returns the reloaded
    model. Raises AssertionError if any count differs - this is a real failure, not a
    discrepancy to log, since it means the serializer silently dropped data.
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    serializer = PatchedProteusSerializer()
    serializer.save(model, dir_path, filename)
    reloaded = serializer.load(dir_path, filename)

    original_cm = model.conceptualModel
    reloaded_cm = reloaded.conceptualModel

    checks = {
        "taggedPlantItems": (
            len(original_cm.taggedPlantItems),
            len(reloaded_cm.taggedPlantItems),
        ),
        "pipingNetworkSystems": (
            len(original_cm.pipingNetworkSystems),
            len(reloaded_cm.pipingNetworkSystems),
        ),
        "processInstrumentationFunctions": (
            len(original_cm.processInstrumentationFunctions),
            len(reloaded_cm.processInstrumentationFunctions),
        ),
    }
    for field_name, (before, after) in checks.items():
        assert before == after, (
            f"round-trip count mismatch for {field_name}: {before} before, {after} after"
        )

    return reloaded
