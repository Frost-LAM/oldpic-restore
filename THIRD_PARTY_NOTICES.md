# Third-party licenses

This tool downloads model weights on first run and does not vendor them.
The inference code needed to load those weights lives under
`old_photo_restore/third_party/` and keeps the upstream license.

**CodeFormer and the InsightFace model packs are not licensed for
commercial use.** The rest of this list is not a grant to ignore that.

## CodeFormer (network code and `codeformer.pth`)

- Project: https://github.com/sczhou/CodeFormer
- Code license: **S-Lab License 1.0** (Copyright 2022 S-Lab). Non-commercial
  redistribution and use only. Commercial use requires contacting the
  contributors. Full text: [LICENSES/S-Lab-1.0.txt](LICENSES/S-Lab-1.0.txt).
- Weights: `codeformer.pth` from the v0.1.0 GitHub release (the `params_ema`
  checkpoint). The CodeFormer project states that redistribution and use of
  the project follow the S-Lab License 1.0. Treat the weights the same way.
- The VQGAN blocks upstream were adapted from
  [Unleashing Transformers](https://github.com/samb-t/unleashing-transformers).
- Local file: `old_photo_restore/third_party/codeformer.py`.

## NAFNet (network code and `NAFNet-REDS-width64.pth`)

- Project: https://github.com/megvii-research/NAFNet
- Code license: **MIT**, Copyright (c) 2022 megvii-model. Portions derived
  from BasicSR are **Apache-2.0**, Copyright 2018-2020 BasicSR Authors.
- Weights: NAFNet-REDS-width64, published by the NAFNet authors (Google Drive
  / Baidu in the upstream docs). This tool downloads a commonly mirrored copy
  of that file from `nyanko7/nafnet-models` on Hugging Face. The NAFNet
  repository does not publish a separate weight license; the repository
  itself is MIT. SHA-256 of the file this tool expects:
  `175fe8b3cdf3abedfbc87769779c3d9f491e05bb2e73ea9d627883f90a4b2df3`.
- Local file: `old_photo_restore/third_party/nafnet.py`.
- The GoPro-trained NAFNet checkpoint is intentionally not the default.

## Real-ESRGAN (`RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth`)

- Project: https://github.com/xinntao/Real-ESRGAN
- Code license: **BSD-3-Clause**, Copyright (c) 2021, Xintao Wang. The RRDB
  blocks also descend from ESRGAN / BasicSR (Apache-2.0).
- Weights: released as GitHub release assets in the same repository
  (`v0.2.1` for x2plus, `v0.1.0` for x4plus). No separate weight license is
  published. The prevailing reading, which this notice does not claim the
  author has confirmed, is that the BSD-3-Clause terms cover those files.
  Keep the copyright notice, and do not use the author's name to endorse a
  derived product.
- Training data includes DIV2K, whose own terms restrict some uses to
  academic research. Whether that restriction attaches to the trained weights
  is an open question; it is recorded here so it is not hidden.
- Local file: `old_photo_restore/third_party/rrdbnet.py`.

## ParseNet (`parsing_parsenet.pth`)

- Code: the ParseNet definition shipped by
  [facexlib](https://github.com/xinntao/facexlib) (MIT; the facexlib
  collection asks you to follow each algorithm's original license). The
  header in facexlib says the file is modified from
  [PSFRGAN](https://github.com/chaofengc/PSFRGAN).
- Weights: `parsing_parsenet.pth` from the facexlib v0.2.2 release
  (`https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth`).
  facexlib documents this file as the ParseNet face-parsing model (19 CelebAMask-HQ
  classes). No additional commercial restriction is stated on that release
  beyond the underlying parsing-network license.
- Local file: `old_photo_restore/third_party/parsenet.py`.
- Class order used by the blend mask (background, skin, brows, eyes, glasses,
  ears, earring, nose, mouth, lips, neck, necklace, cloth, hair, hat) is the
  order this checkpoint emits. Mouth and skin are not used as reference-detail
  donors.

## InsightFace: SCRFD, ArcFace (`w600k_r50`), glintr100

- Code project: https://github.com/deepinsight/insightface
- **Code** license: MIT. Academic and commercial use of the library code is
  allowed.
- **Model weights are non-commercial research only.** Upstream states that
  the training data (and the models trained with it) are available for
  non-commercial research purposes only, and that both manually downloaded
  and auto-downloaded models follow that policy.
- Packs this tool downloads:
  - `buffalo_l` (v0.7 release): `det_10g.onnx` is SCRFD-10GF, used for
    detection; `w600k_r50.onnx` is a ResNet-50 ArcFace model trained on
    WebFace600K, used for the identity score and for matching references.
  - `antelopev2` (v0.7 release), only when reference photos are used:
    `glintr100.onnx` is a ResNet-100 recogniser trained on Glint360K. It is
    a second opinion on reference matches and on the reference-parameter
    search. It is not used to change pixels by itself.
- Upstream commercial-licensing note dated 2025-11-24: for the open-source
  recognition packs (buffalo_l is the example they give), contact
  `recognition-oss-pack@insightface.ai`.
- This repository does not vendor the ONNX files.

## Detection note

The original prototype detected faces with facexlib's RetinaFace-ResNet50
(PyTorch, MIT upstream). This package uses SCRFD-10GF from the same
InsightFace pack as ArcFace so detection stays on ONNX Runtime, on CPU, and
can be tiled over a large group photo without loading a second PyTorch
detector. Both return a box and five landmarks.

## Runtime libraries

PyTorch, ONNX Runtime, OpenCV, NumPy, SciPy, Pillow, PyYAML, and tqdm are
installed from their own distributions and are not vendored here. PyTorch
and ONNX Runtime are Apache-2.0; OpenCV is Apache-2.0. Install the CPU build
of PyTorch. This tool never selects a CUDA or MPS device.
