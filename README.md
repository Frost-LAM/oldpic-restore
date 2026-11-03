# old-photo-restore

[中文说明](README.zh-CN.md)

Restore a blurry, camera-shaken, or old group photo to a sharper image **without changing who the people are**. The pipeline is a non-generative deblur, then a high-fidelity face restore blended back toward the original, then a light super-resolution mix. It is not a beauty filter and it does not invent a new face.

It is built for one photo with any number of faces, from a single portrait to a large group photo. `--device auto` (the default) uses Apple Silicon MPS, then CUDA if that is what the machine has, otherwise CPU. Models are loaded one at a time and the heavy stages run in tiles. Half-scale deblur stays the default on every device.

**License, read this first.** The pipeline code is MIT. **CodeFormer (the face stage) is under the S-Lab License 1.0 and is non-commercial.** The InsightFace weights used for detection and identity (ArcFace `w600k_r50`, and `glintr100` when you pass references) are for **non-commercial research** only. Details and the other model terms are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Weights are downloaded on first run and are not in this repository.

## What it does

1. **Keep the original as the colour and identity reference.** The photo you pass in is never overwritten.
2. **Deblur without a generative model.** [NAFNet-REDS-width64](https://github.com/megvii-research/NAFNet) runs in tiles on a half-resolution copy (downscale, deblur, continue from there). NAFNet tends to darken the frame by a few levels, so the low-frequency colour of the original is put back.
3. **Find every face and number it left to right** (`P1` … `Pn`). A labelled overview is saved so you can talk about faces by number.
4. **Restore each face with CodeFormer at fidelity `w=0.9`.** Each result is colour-matched to the crop it came from, then blended back at 70%. Weak or low-confidence faces are blended at about 55% (50% if the identity score is very low). You can override any face in the config.
5. **Upscale the scene with Real-ESRGAN, mixed 75/25 with Lanczos** so the background does not look painted. `x2plus` brings the half-resolution image back to the original size. `x4plus` on that same half-resolution image produces a 2× output. Restored faces are pasted on after the upscale, so Real-ESRGAN does not repaint them.
6. **Score identity.** For every face, ArcFace cosine similarity is measured between the original crop and the restored crop. The report lists the score, and a weak score lowers that face's blend automatically.
7. **Optional references.** If you supply other photos of the same people — including photos from a different age — each reference face is matched to a numbered face by ArcFace. `glintr100` is a second opinion and can reject an ambiguous match. You can also pin the mapping in a YAML file. From a matched reference the tool takes **only high-frequency detail from the brows, eyes, glasses, and nose**. It does not copy the mouth or the skin. That detail is injected lightly into CodeFormer's input, with a further ~0.3 detail boost, and the face is blended back around 50–60%. About 36 parameter combinations are scored per face against both the original and the reference; the identity guard drops any candidate that drifts away from the person in the photo. If the chosen pass changes teeth or lips, the mouth is put back from the non-reference result.

These are the defaults because they are the settings that held up on a large group photo. The following were tried and are **not** the defaults: NAFNet-GoPro, deblurring at full resolution, a second deblur pass, Richardson–Lucy kernel estimation (it fails on blown highlights and on blur that changes across the frame), and RestorerID / InstantRestore (too heavy for CPU). A manual eye/iris patch was also tried and removed: it made no visible difference, so this tool has no local patch step.

## Install

Python 3.10 or newer. 3.11 or 3.12 is a good choice on an M-series Mac.

### Mac (Apple Silicon)

M1, M2, M3, and M4. The setup script creates a virtualenv, installs the PyPI PyTorch wheel (MPS), and `onnxruntime` (the macOS arm64 build includes the CoreML provider).

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
old-photo-restore photo.jpg -o out/
```

That is the same as `pip install -r requirements-mac.txt` after `python3 -m venv .venv`. The script refuses to run on Intel Macs and on Linux. Check the install with:

```bash
python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.backends.mps.is_available(), ort.get_available_providers())"
```

`mps available: True` and `CoreMLExecutionProvider` in the provider list is the expected result. The default `--device auto` then runs NAFNet, CodeFormer, the face parser, and Real-ESRGAN on MPS in fp32. ArcFace may use CoreML. The face detector stays on CPU: SCRFD's dynamic outputs make the CoreML provider abort with a rank error and a long log. That CPU pass is about two seconds. Face restoration is still one face at a time.

If `git clone` of a private checkout fails with `Invalid username or token`, macOS Keychain is holding an old GitHub credential. Run `gh auth login`, or create a personal access token and use it as the password. The repository files are not the problem.

Slow Hugging Face downloads (a few MB/s, common in mainland China) can use a mirror. `OPR_HF_ENDPOINT` wins over `HF_ENDPOINT`. Only `huggingface.co` URLs are rewritten; GitHub release files are unchanged. Partial downloads are kept and resumed.

```bash
export OPR_HF_ENDPOINT=https://hf-mirror.com
old-photo-restore --download-weights
```

`CERTIFICATE_VERIFY_FAILED` from a python.org Python on macOS means the installer did not wire up certificates. This package downloads through `certifi`. If it still fails, run `/Applications/Python 3.12/Install Certificates.command` (the version number matches the Python you installed) and try again.

Tiles are larger than the CPU profile (deblur 512, Real-ESRGAN 384) because the Mac's memory is unified. Half-scale deblur stays the default. On a 16 GB or larger Mac you can try full-scale deblur with `--deblur-scale 1`; it uses more memory and is not the setting that was tuned on a large group photo.

### Linux

CPU, which is what `scripts/setup_linux.sh` installs. The CPU PyTorch index matters: a plain `pip install torch` on Linux often installs a large CUDA build.

```bash
bash scripts/setup_linux.sh
source .venv/bin/activate
old-photo-restore photo.jpg -o out/ --device cpu
```

The packages after torch are listed in `requirements-linux.txt`. CPU tiles stay at deblur 384 and Real-ESRGAN 256, which fits a machine with a few gigabytes free.

NVIDIA GPU, optional: install `requirements-linux-cuda.txt` (PyPI torch plus `onnxruntime-gpu`) instead of the CPU torch line, then run with `--device cuda`. Do not mix `onnxruntime` and `onnxruntime-gpu` in one environment.

Weights, on every platform, are cached in `~/.cache/old-photo-restore/weights` (or `$OPR_WEIGHTS_DIR`, or `--weights-dir`). The first run with no references downloads roughly 1.2 GB (NAFNet, CodeFormer, ParseNet, both Real-ESRGAN nets, and the InsightFace `buffalo_l` pack). Reference mode also downloads the `antelopev2` pack for `glintr100` (a few hundred MB more).

```bash
old-photo-restore --download-weights
```

## Usage

```bash
old-photo-restore photo.jpg -o out/
old-photo-restore photo.jpg -o out/ --scale both --blend 0.7
old-photo-restore photo.jpg -o out/ --refs refs/ --ref-map refs.yaml
```

| Flag | Meaning |
| --- | --- |
| `-o, --output` | Directory for the images and `report.txt` |
| `--scale 1\|2\|both` | `1` original size, `2` double size, `both` (default) |
| `--blend` | How much of the restored face replaces the photo. `0.7` default, `0` keeps the photo |
| `--fidelity` | CodeFormer `w` in `[0, 1]`. `0.9` default (higher stays closer to the person) |
| `--refs` | Directory of reference photos |
| `--ref-map` | YAML file, reference filename → face number |
| `--deblur-scale` | Default `0.5`. `1` deblurs at full size and is not recommended |
| `--esrgan-mix` | Real-ESRGAN share of the upscale. Default `0.75`, the rest is Lanczos |
| `--config` | YAML merged on top of the packaged defaults |
| `--device auto\|mps\|cuda\|cpu` | Default `auto`: MPS on Apple Silicon, else CUDA, else CPU |
| `--max-faces` | Restore only the N largest faces. Numbering still covers everyone |
| `--cpu-threads` | Threads for CPU work and ONNX. Default `min(8, core count)` |
| `--weights-dir` | Weight cache |
| `--download-weights` | Fetch every weight, including glintr100, and exit |

Packaged defaults live in [`old_photo_restore/default.yaml`](old_photo_restore/default.yaml). Copy that file, edit it, and pass `--config`. Per-face blends:

```yaml
blend:
  per_face:
    2: 0.55
    8: 0.60
```

Face numbers match the overview (`P2` is `2`). A value here wins over the automatic weak-face rule.

### References, including a different age

Put one or more photos of the same people in a directory. A reference can be a portrait or a group, and it can be years away from the photo you are restoring: the matcher compares identity embeddings, not age or expression. Each detected reference face is assigned to at most one numbered face. A match needs an ArcFace cosine of at least `0.22` (different-age pairs are often well below a same-day threshold) and a small margin over the runner-up. `glintr100` vetoes the match only when it clearly prefers someone else.

Or pin it yourself. [`examples/refs.example.yaml`](examples/refs.example.yaml):

```yaml
alice-2010.jpg: 2
bob.png: 8
group-photo.jpg#1: 11   # 0-based index, largest face first
```

`#k` is only needed when a reference file contains more than one person. Without it, the largest face in that file is used.

What is copied from the reference is narrow on purpose. The reference is warped onto the photo face's landmarks, then only the high-frequency band inside the brows, eyes, glasses, and nose is eligible. Mouth, teeth, and skin stay with the photo. If a candidate still changes the mouth, that region is restored from the CodeFormer result that did not see the reference.

## Outputs

| File | What it is |
| --- | --- |
| `restored.png` | Original resolution. Real-ESRGAN x2plus on the half-size image, mixed with Lanczos, faces pasted on top |
| `restored-2x.png` | Twice the original. Real-ESRGAN x4plus on the half-size image, same mix, faces pasted |
| `faces-overview.jpg` | The photo with `P1`…`Pn` drawn on it (long side capped so the labels stay readable). Coordinates in `report.txt` are in the original pixels |
| `compare-side-by-side.jpg` | Full frame, original next to the restored original-size image |
| `compare-faces.jpg` | One pair per restored face: original crop, restored crop |
| `compare-refs.jpg` | Only if a reference was applied: original, restored, warped reference |
| `report.txt` | Settings, per-face detection and identity scores, weak faces, reference choices, timings |

Identity lines look like this:

```text
P3  det=0.982  blend=0.70  reason=default  identity_deblur=0.61  identity_full=0.58  identity_out=0.66
```

`identity_deblur` is the deblurred crop against the original, `identity_full` is CodeFormer before blending, and `identity_out` is the face that was actually pasted. The blend drops when the full restore falls away from the deblur-only score, when the absolute score is low, or when the detector is unsure. Those faces are listed on the `weak faces:` line.

## Photoshop

Photoshop 2026 no longer has Shake Reduction. The nearest manual alternative is Filter > Sharpen > Smart Sharpen with Remove: Motion Blur, Angle 0, Amount 100, Radius 4, Reduce Noise 15. That is a single global sharpen. It does not replace the face-by-face identity check in this tool.

## Runtimes

These are example timings from representative runs. `report.txt` records the seconds for your own run.

On a 4-core Linux CPU, a 960×695 group photo with two reference-guided faces took about 2.5 minutes. On an Apple Silicon Mac (M-series), a single-face test photo (weights already cached) took 55 seconds on MPS — deblur 7.8s, detect 1.9s, the face 1.5s, and about 40s to write both the original-size and 2× files — and 104 seconds on CPU when only the original size was requested. A large group photo is still mostly the two upscales, so expect longer than that one-face run. The detector is CPU even on a Mac; ArcFace may use CoreML.

## Troubleshooting (Apple Silicon)

- **`mps available: False`.** The torch wheel is not the macOS arm64 build. Re-run `bash scripts/setup_mac.sh` with an arm64 `python3` (3.11 or 3.12).
- **`NotImplementedError` / "not currently implemented for the MPS device".** The tool sets `PYTORCH_ENABLE_MPS_FALLBACK=1` before importing PyTorch, and if one module still fails it moves that module to CPU and continues. To skip MPS entirely: `--device cpu`.
- **`MPS backend out of memory`.** Lower `deblur.tile` and `upscale.tile` in a config file (try 384 and 256), or pass `--device cpu`. Half-scale deblur is already the default; `--deblur-scale 1` is the option that uses more memory.
- **CoreML errors during ArcFace.** The detector does not use CoreML. If ArcFace's CoreML run fails, that session retries on CPU. A wrong or empty face list is a reason to run `--device cpu` and keep the log.
- **Black or NaN output.** Run once with `--device cpu`. The PyTorch stages are forced to fp32 because several MPS kernels reject fp16; a black frame means something else failed.

## Limitations

- It is slow in proportion to the number of faces. Each face is one CodeFormer pass at 512² (the batch size is 1 on every device). A reference search is about six passes per matched face (36 scored mixes). A large group photo is a long run, not a preview.
- CodeFormer is still a learned face prior. Fidelity `0.9` and the blend limit how far it may move; they are not a proof that every pixel of identity survived. Read `report.txt`.
- The default face threshold is `0.60`. At `0.40`, decorative carvings or patterns were detected (scores around 0.45 and 0.49) and CodeFormer painted a face into them. A five-point geometry check rejects landmarks that are not eyes-above-nose-above-mouth. For a crowded, blurry group where real faces are dropped, lower the threshold in a config file and pass `--config`:

  ```yaml
  detect:
    threshold: 0.45
  ```
- Blown lights and blur that changes across the frame are why a global deconvolution kernel is not used. Those regions stay imperfect.
- A reference from another age can put real texture back into brows, eyes, glasses, and the nose. It will not change expression, and it is not allowed to replace the mouth.
- There is no beautification, makeup, or age edit, and no manual iris patch.
- If a tile runs out of memory, lower `deblur.tile` and `upscale.tile` in the config (CPU defaults are 384 and 256; MPS defaults are 512 and 384) and run again. Weights stay cached.

## Tests

```bash
pytest                 # unit tests, no downloads
pytest -m smoke        # CPU end-to-end on a public-domain group photo; downloads weights
```

The smoke test uses the 1927 Solvay conference photograph (public domain). It does not use or store personal photos.

## Layout

`old_photo_restore/pipeline.py` is the stage order. `third_party/` holds the network definitions required to load the published weights, each with its own license header. `default.yaml` is the knob list.
