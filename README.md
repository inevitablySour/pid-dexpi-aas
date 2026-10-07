# P&ID -> DEXPI -> AAS

Turns P&ID drawings into DEXPI models and Asset Administration Shells (IDTA 02012-1-0).
Input: a draw.io file, a picture of a P&ID, or both. You check and fix the result in a review
app, then download the DEXPI file and the AAS. Part of the INCEPTION/CHILL research minor;
project background in [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Quick start

```bash
git clone https://github.com/inevitablySour/pid-dexpi-aas.git && cd pid-dexpi-aas
./launch_app.sh              # installs and opens the app - draw.io files work straight away
./scripts/get_drawio.sh      # Linux: the draw.io app, for line routes, tees and the flow map
                             # (macOS / Windows: install draw.io normally)
```

For pictures (image mode), also install the model environment and download the model:

```bash
./ml/pid2graph/setup.sh              # with an NVIDIA GPU: GPU build; without: CPU (minutes per picture)
gh auth login && ./ml/get_model.sh   # trained model -> ml/models/
```

`setup.sh` is a bash script; on Windows run it in WSL2. Training needs the CUDA toolkit;
running the model only needs the normal NVIDIA driver.

## Using the app

Upload in the sidebar:

| Input | Where items and lines come from |
|---|---|
| draw.io file | the file itself |
| picture | the trained model, OCR for the tags, and a pipe tracer that follows the drawn lines |
| both | the drawing is the truth; the picture checks it (the Compare tab shows where they differ) |

Click anything on the map to edit it in the side panel: tag, DEXPI type, connections,
streams in / out, and DEXPI specifications for items and pipes. Draw a box with **Add item**
for anything that was missed. Then **Build & validate DEXPI** and download the DEXPI file and
the `.aasx`. A review can be saved and resumed later.

## Sharing the app

```bash
./scripts/serve_public.sh            # prints a public https link and a password
./scripts/serve_public.sh --reload   # after a code change: restart the app, same link
```

A free Cloudflare tunnel; the app stays on this PC behind the password. Logins last 48 hours
of inactivity. The link changes when the script restarts.

## Command line

```bash
# one drawing -> DEXPI XML + a re-drawn SVG
.venv/bin/python scripts/run_pipeline.py "drawing.drawio" out/

# several drawings of one plant -> one AAS
.venv/bin/python -m pid_dexpi_aas.aas.export subprojects/Lumen/*.drawio -o lumen.aasx \
    --process-plant-name "Lumen gramflow setups" --aas-version 3.0
```

AAS 3.0 is checked with the IDTA's test engine; 3.1 (`--aas-version 3.1`) with aas-core3.1.

## Project layout

```
app.py                  the review app
src/pid_dexpi_aas/      the converter
  review.py               drawing / picture -> review rows -> DEXPI
  drawio/                 reading draw.io files
  dexpi/                  building and exporting the DEXPI model
  render/                 the flow map and the re-drawn SVG
  image/                  OCR and picture helpers
  aas/                    DEXPI -> AAS
ml/                     the symbol model - see ml/README.md
scripts/                setup and sharing helpers, command-line pipeline
tests/                  tests (.venv/bin/python -m pytest tests)
docs/                   project background, pyDEXPI notes, DEXPI class lists
subprojects/            the team's drawings
```

pyDEXPI 1.2.0 cannot export piping or instrumentation, so this project patches its exporter;
it is pinned for that reason. Details and the other workarounds:
[`docs/pydexpi-notes.md`](docs/pydexpi-notes.md).
