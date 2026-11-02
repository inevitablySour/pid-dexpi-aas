# P&ID -> DEXPI -> AAS

Turns P&ID drawings into DEXPI models and Asset Administration Shells (IDTA 02012-1-0).
Input: a draw.io file, a picture of a P&ID, or both. You check and fix the result in a review
app, then download the DEXPI file and the AAS. Part of the Green Skills by Integrated Infrastructure
project (Zuyd University of Applied Sciences); background in [`docs/HANDOFF.md`](docs/HANDOFF.md).

## Install and run on your own computer

The app runs locally in your browser. Two parts:

- **The app** (draw.io files, the editor, DEXPI and AAS export). Works on Linux, macOS and Windows.
- **The picture model** (reading P&IDs from images). Linux, or Windows through WSL2; on macOS only
  on the CPU and untested. Optional: without it everything except picture upload works.

Needs **Python 3.10 or newer** and **git**. The app opens at **http://localhost:8501** and only
accepts connections from your own computer (`.streamlit/config.toml`).

### Linux (Ubuntu, Mint, Debian)

```bash
sudo apt install python3 python3-venv git
git clone https://github.com/inevitablySour/pid-dexpi-aas.git
cd pid-dexpi-aas
./launch_app.sh            # first run installs everything (a minute), then opens the browser
./scripts/get_drawio.sh    # optional: draw.io, for line routes, tees, the flow map and the Editor tab
```

### macOS

```bash
brew install python git            # Homebrew: https://brew.sh
brew install --cask drawio         # optional: line routes, tees, the flow map
git clone https://github.com/inevitablySour/pid-dexpi-aas.git
cd pid-dexpi-aas
./launch_app.sh
.venv/bin/python scripts/get_drawio_editor.py   # optional: the Editor tab
```

### Windows (PowerShell)

```powershell
winget install Python.Python.3.12 Git.Git
winget install JGraph.Draw                       # optional: line routes, tees, the flow map
# close and reopen PowerShell, then:
git clone https://github.com/inevitablySour/pid-dexpi-aas.git
cd pid-dexpi-aas
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python scripts\get_drawio_editor.py   # optional: the Editor tab
.venv\Scripts\streamlit run app.py
```

Next time: `cd pid-dexpi-aas` and `.venv\Scripts\streamlit run app.py`
(Linux / macOS: `./launch_app.sh`). Stop the app with Ctrl+C in that window.

### Cairo (optional: the re-drawn P&ID)

The check sheet and the build page show the P&ID re-drawn from the DEXPI file. That drawing
uses **Cairo**, a graphics library installed on the computer itself (not with pip). Without it
everything works except that one picture, which is replaced by a note.

| OS | Install |
|---|---|
| Linux | usually already there; else `sudo apt install libcairo2` |
| macOS | `brew install cairo` |
| Windows | install the *GTK for Windows Runtime* (github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer), then reopen PowerShell; or skip it |

### The picture model (optional)

```bash
./ml/pid2graph/setup.sh              # with an NVIDIA GPU: GPU build; without: CPU (minutes per picture)
gh auth login && ./ml/get_model.sh   # the trained model -> ml/models/ (GitHub CLI: https://cli.github.com)
```

`setup.sh` is a bash script: on Windows, install WSL2 (`wsl --install` in an administrator
PowerShell), open the Ubuntu terminal, and follow the Linux steps there, picture model included;
the app then opens in your Windows browser at http://localhost:8501. Running the model needs only
the normal NVIDIA driver; training also needs the CUDA toolkit.

### If something goes wrong

| Problem | Fix |
|---|---|
| `python3: command not found` / `py` not found | Install Python 3.10+ (above) and open a new terminal |
| Browser does not open | Open http://localhost:8501 yourself |
| Port 8501 in use | `./launch_app.sh --server.port 8502` (Windows: add `--server.port 8502`) |
| "ML environment not found" | Run the picture model steps, or use draw.io files only |
| "The re-drawn P&ID needs the Cairo graphics library" | Install Cairo (above), or ignore it |
| Editor tab missing | Run `scripts/get_drawio_editor.py` with the app's Python (above) |

## Using the app

Upload in the sidebar:

| Input | Where items and lines come from |
|---|---|
| draw.io file | the file itself |
| picture | the trained model, OCR for the tags, and a pipe tracer that follows the drawn lines |
| both | the drawing is the truth; the picture checks it (the Compare tab shows where they differ) |
| Editor tab | draw.io itself, in the page: draw a P&ID or change the uploaded one; Save loads it. Group (Ctrl+G) several shapes to make them one DEXPI item. **Guided check** in its panel: every item one by one (lines hidden), then every uncertain connection (model-only, or a guessed draw.io end) - or keep only the sure lines, or start with none |
| Training data | "Add this drawing to the training data": one button (or download, correct in draw.io, add in the sidebar); retrains automatically at night when enough is added (ml/README.md) |
| Plant AAS | several drawings in one AAS: build each, "Add this drawing to the plant AAS", download once; a tag on several drawings is one asset; plant sections (when frames are used as sections) as an IDTA 02011 hierarchy |
| Spec sheet | sidebar "Specifications": download an Excel template (blank, or with this drawing's tags), fill in tag + specifications, upload it - matching items get the values automatically (empty fields only) after a draw.io file or a picture is read; the editor's "Sync from the spec sheet" overwrites |

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

## Privacy and security

- **Uploads are kept only while you use them.** A picture, a draw.io file and the model's result are
  stored in a private folder for your browser session (readable only by the app's user account,
  random name). Removing or replacing the file deletes its copies at once; a session left open or
  closed without removing is deleted after 3 hours. Nothing is shared between sessions.
- **Nothing leaves the computer.** The model runs on the same machine as the app; no picture is
  sent to an online service.
- **Encrypted when shared.** Run locally, the app only listens on your own computer. Shared with
  `serve_public.sh`, visitors connect over HTTPS (encrypted), behind a password. The free
  Cloudflare tunnel decrypts traffic at Cloudflare before passing it on encrypted; for drawings
  that must not pass a third party, share over a private network (e.g. Tailscale) instead.
- **Kept on purpose:** a drawing you add with "Add this drawing to the training data" is stored as
  training data (`data/PID2Graph`), because that is what the button is for.

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
