# Concrete defect recognition: live demo website

Upload a photo of concrete. Our **own CNN** and the fine-tuned **ResNet-18** say whether they see
**crack, rust or spalling**, side by side. Everything runs **inside the browser** (ONNX Runtime Web).
There is no server-side computation, the photo never leaves the computer, and no internet is needed.

## Start it (offline, on your laptop)

1. Double-click **`start_demo.bat`**. A black window opens and the browser shows the demo at <http://localhost:8000/>.
2. Keep the black window open while you use the demo. Close it to stop.

(Without the .bat file: `python serve.py` in this folder. Double-clicking `index.html` does **not** work,
because browsers do not allow a page opened from the disk to load the network files.)

Useful addresses:
- `http://localhost:8000/?example=6` opens example 6 directly (1–8, in the order shown on the page).
- `http://localhost:8000/?mode=synthetic` starts in the synthetic task.

## What it shows

| Task | Networks | Output |
|---|---|---|
| Real bridge photos (default) | ResNet-18 (ImageNet → synthetic → real, test mAP 0.746) and own CNN (synthetic → real, mAP 0.704) | a probability per label. A label is *detected* when it passes the threshold tuned on real_val (tick on the bar): ResNet 0.35 / 0.45 / 0.30, own CNN 0.25 / 0.35 / 0.30 for crack / rust / spalling |
| Synthetic task | ResNet-18 (synthetic test accuracy 0.866) and own CNN (0.853), trained only on synthetic images | one of pristine / crack / rust / spalling |

The 8 example photos are dacl10k **test** photos, chosen by their annotated label with a fixed random seed,
**not** by whether the networks get them right. The page shows the dacl10k annotation next to the result.

The photo is prepared exactly as in training: the whole photo is squeezed to 256 × 256 (bicubic, like Pillow),
then the centre 224 × 224 is taken. The page shows this "what the network sees" image.

## Ideas for the class

- Click the examples. Some are right and some are wrong (for example, the "rust + spalling" photo gets "crack + spalling").
  That is honest and leads to the discussion of the scores.
- Switch to **Synthetic task** and click a real photo. Networks trained only on synthetic images are often wrong
  on real photos: this is the domain gap.
- Upload a phone photo of a wall or bridge (or paste one with Ctrl + V).

## Checks that were done (22 Nov 2026)

- The ONNX files in Python on all 975 real test photos: ResNet-18 mAP 0.7460 and own CNN 0.7038, the same as in the model cards.
- Browser (Microsoft Edge) vs Python on the 8 examples, all 4 networks: **the same decision in 32 of 32 cases**
  (detected labels or top class). Probabilities differ by at most 0.05 for three networks, and by 0.32 for the
  synthetic own CNN on one real photo, with the same top class. The reason: Python reads JPEGs with Pillow's
  reduced-size decoding, while the browser decodes at full size and then shrinks.

## Online (Netlify)

The folder is a plain static website (about 115 MB, the largest file 45 MB). It is hosted from the GitHub
repository <https://github.com/Bikalpa-gautam/SciML-Demo> with Netlify:

1. On netlify.com: **Add new site → Import an existing project → GitHub** → choose `SciML-Demo`.
2. Build settings: **leave the build command empty**, publish directory **`.`** (already set in `netlify.toml`). Deploy.
3. Every new push to `main` updates the site automatically.

`serve.py` and `start_demo.bat` are only needed offline; Netlify serves the files itself.
The supervisor agreed on 22 Nov 2026 to publish the demo with the trained networks.

## Licences

- **Code** (`index.html`, `style.css`, `app.js`, `serve.py`, `start_demo.bat`): MIT (see `LICENSE`).
- **Example photos** (`examples/`): from dacl10k (Flotzinger et al., WACV 2024), licence **CC BY-NC 4.0**: credit the authors, non-commercial use only. Not covered by the MIT licence.
- **Networks** (`models/`): trained by Bikalpa Gautam on the ETH course's synthetic images and on dacl10k photos; published with the supervisor's agreement for non-commercial teaching and research use.
- **ONNX Runtime Web** (`ort/`): MIT licence, © Microsoft Corporation.

## Files

| File | What it is |
|---|---|
| `index.html`, `style.css`, `app.js` | the page (plain HTML / CSS / JavaScript, no build step) |
| `models/*.onnx` | the 4 networks, copied from `presentation/models/` (exported 25 Oct 2026) |
| `ort/` | ONNX Runtime Web 1.20.1 (MIT licence, Microsoft), the engine that runs `.onnx` files in the browser |
| `examples/` | 8 dacl10k test photos (resized to at most 1280 px) and `examples.json` with their annotations |
| `serve.py`, `start_demo.bat` | the small local web server for offline use |
| `netlify.toml` | Netlify settings (no build step, correct file types for `.wasm` / `.mjs`) |
