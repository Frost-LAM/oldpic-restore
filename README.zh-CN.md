# old-photo-restore

[English](README.md)

## 中国大陆：先设下载镜像

huggingface.co 在国内常常只有 2–3 MB/s，第一次下载大约要 15 分钟。把 Hugging Face 换成镜像（例如 hf-mirror.com）即可。只改写 `huggingface.co` 的地址，GitHub 上的权重不受影响。中断后的半成品会留下来，下次接着下。`OPR_HF_ENDPOINT` 优先于 `HF_ENDPOINT`。

```bash
export OPR_HF_ENDPOINT=https://hf-mirror.com
old-photo-restore --download-weights
```

python.org 安装的 Python 在 macOS 上下载权重时如果报 `CERTIFICATE_VERIFY_FAILED`，本程序已经改用 certifi 的证书。若仍然失败，运行 `/Applications/Python 3.12/Install Certificates.command`（版本号换成你安装的 Python），然后再试。

私人仓库 `git clone` 报 `Invalid username or token` 时，多半是 macOS 钥匙串里留着过期的 GitHub 凭据。运行 `gh auth login`，或换一个个人访问令牌（PAT）。仓库本身没有坏。

把模糊、手抖或老旧的合影修清晰，**同时不改变照片里的人是谁**。流程是：先做非生成式去模糊，再以高保真方式修复人脸并按比例叠回原图，最后做一次较轻的超分辨率混合。它不是美颜，也不会另造一张脸。

一张照片里可以有任意多张脸，从单人肖像到大合影都可以。`--device auto`（默认）会优先用 Apple 芯片的 MPS，其次是 CUDA，否则用 CPU。模型一次只加载一个，耗内存的步骤按图块处理。无论哪台机器，去模糊默认都在半分辨率上做。

**请先看许可。** 本仓库的流程代码是 MIT。**人脸修复所用的 CodeFormer 使用 S-Lab License 1.0，仅限非商业用途。** 用于检测和身份比对的 InsightFace 权重（ArcFace `w600k_r50`，以及在提供参考图时使用的 `glintr100`）**仅限非商业研究**。其余模型条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。权重在第一次运行时下载，不放进仓库。

## 它做什么

1. **原图始终是色彩和身份的参照。** 输入的照片不会被覆盖。
2. **非生成式去模糊。** [NAFNet-REDS-width64](https://github.com/megvii-research/NAFNet) 在半分辨率上分块运行（先缩小、去模糊，后续都基于这张半尺寸图）。NAFNet 会把画面压暗大约几个色阶，因此会把原图的低频色彩补回去。
3. **找出所有人脸，从左到右编号**（`P1` … `Pn`），并保存一张带编号的总览，方便按编号指认。
4. **用 CodeFormer 以保真度 `w=0.9` 修复每张脸。** 修复结果先与修复前的裁块做色彩匹配，再以 70% 的强度叠回去。偏弱或检测置信度低的脸用大约 55%（身份分非常低时用 50%）。也可以在配置里单独指定某张脸。
5. **用 Real-ESRGAN 放大场景，并与 Lanczos 按 75/25 混合**，减轻“画出来”的感觉。半尺寸图用 `x2plus` 得到原始分辨率；同一张半尺寸图用 `x4plus` 得到 2 倍输出。人脸是在放大之后贴回去的，避免 Real-ESRGAN 把脸再画一遍。
6. **身份自检。** 每张脸计算原图裁块与修复结果之间的 ArcFace 余弦相似度。报告里会列出分数；分数偏低时，这张脸的混合比例会自动降低。
7. **可选参考图。** 如果你另外提供同一个人的照片——年龄不同也可以——每张参考脸会用 ArcFace 自动对上编号。`glintr100` 作为第二意见，可以否决含糊的匹配。也可以用 YAML 明确指定对应关系。从匹配上的参考图里，工具**只取眉毛、眼睛、眼镜和鼻子的高频细节**，不取嘴巴，也不取皮肤。这些细节以较低强度注入 CodeFormer 的输入，并再加大约 0.3 的细节增强，混合比例大约在 50%–60%。每张脸大约搜索 36 组参数，同时对照原图和参考图打分；偏离照片中这个人的候选会被丢掉。如果某一次结果改动了牙齿或嘴唇，嘴部会从“没有使用参考图”的结果里补回来。

这些默认值来自一张大合影上实际有效的设置。下面这些试过，**但不是默认**：NAFNet-GoPro、全分辨率去模糊、去模糊跑两遍、Richardson–Lucy 核估计（高光溢出、以及画面各处模糊程度不一致时会失败）、RestorerID / InstantRestore（CPU 上太重）。也曾试过手动修补眼睛和虹膜，肉眼看不出差别，因此本工具没有任何局部贴片步骤。

## 安装

需要 Python 3.10 或更新版本。M 系列 Mac 上用 3.11 或 3.12 比较合适。

### Mac（Apple 芯片）

适用于 M1、M2、M3、M4。安装脚本会建虚拟环境，安装带 MPS 的 PyPI 版 PyTorch，以及 macOS arm64 上自带 CoreML 执行器的 `onnxruntime`。

```bash
bash scripts/setup_mac.sh
source .venv/bin/activate
old-photo-restore photo.jpg -o out/
```

这和在虚拟环境里执行 `pip install -r requirements-mac.txt` 是一回事。脚本在 Intel Mac 和 Linux 上会拒绝运行。装完可以检查：

```bash
python -c "import torch, onnxruntime as ort; print(torch.__version__, torch.backends.mps.is_available(), ort.get_available_providers())"
```

正常结果是 `mps available: True`，并且提供者列表里有 `CoreMLExecutionProvider`。默认 `--device auto` 会用 fp32 在 MPS 上跑 NAFNet、CodeFormer、人脸解析和 Real-ESRGAN。ArcFace 可以走 CoreML。人脸检测固定用 CPU：SCRFD 的动态输出会让 CoreML 报 rank 错误并刷出很长的日志，而 CPU 检测大约只要两秒。人脸仍然是一张一张处理。

图块比 CPU 配置更大（去模糊 512，Real-ESRGAN 384），因为 Mac 是统一内存。去模糊仍然默认半分辨率。16 GB 或更大内存的机器可以试 `--deblur-scale 1` 做全尺寸去模糊，更吃内存，也不是在大合影上调出来的设置。

### Linux

CPU 安装用 `scripts/setup_linux.sh`。Linux 上直接 `pip install torch` 经常会装上体积很大的 CUDA 版本，所以脚本走 CPU 索引。

```bash
bash scripts/setup_linux.sh
source .venv/bin/activate
old-photo-restore photo.jpg -o out/ --device cpu
```

torch 之后的包在 `requirements-linux.txt`。CPU 的图块仍然是去模糊 384、Real-ESRGAN 256，适合只有几 GB 空闲内存的机器。

NVIDIA GPU（可选）：不要装 CPU 版 torch，改用 `requirements-linux-cuda.txt`（PyPI 的 torch 加上 `onnxruntime-gpu`），然后加 `--device cuda`。不要在同一个环境里同时装 `onnxruntime` 和 `onnxruntime-gpu`。

各平台的权重都缓存在 `~/.cache/old-photo-restore/weights`（也可用 `$OPR_WEIGHTS_DIR` 或 `--weights-dir`）。第一次运行、且不使用参考图时，大约下载 1.2 GB（NAFNet、CodeFormer、ParseNet、两个 Real-ESRGAN 模型，以及 InsightFace 的 `buffalo_l`）。使用参考图时还会下载 `antelopev2`（其中的 `glintr100`，再增加几百 MB）。

```bash
old-photo-restore --download-weights
```

## 用法

```bash
old-photo-restore photo.jpg -o out/
old-photo-restore photo.jpg -o out/ --scale both --blend 0.7
old-photo-restore photo.jpg -o out/ --refs refs/ --ref-map refs.yaml
```

| 参数 | 含义 |
| --- | --- |
| `-o, --output` | 输出目录，里面会有图像和 `report.txt` |
| `--scale 1\|2\|both` | `1` 原始尺寸，`2` 两倍尺寸，`both` 两者都要（默认） |
| `--blend` | 修复后的脸替换原图的比例。默认 `0.7`，`0` 表示完全保留照片 |
| `--fidelity` | CodeFormer 的 `w`，范围 `[0, 1]`。默认 `0.9`（越高越贴近原来的人） |
| `--refs` | 参考照片所在目录 |
| `--ref-map` | YAML：参考图文件名 → 人脸编号 |
| `--deblur-scale` | 默认 `0.5`。设为 `1` 会按全尺寸去模糊，不建议 |
| `--esrgan-mix` | 放大结果里 Real-ESRGAN 的比重。默认 `0.75`，其余是 Lanczos |
| `--config` | 在内置默认值之上再叠一份 YAML |
| `--device auto\|mps\|cuda\|cpu` | 默认 `auto`：Apple 芯片用 MPS，否则 CUDA，再否则 CPU |
| `--max-faces` | 只修复最大的 N 张脸。编号仍然覆盖检测到的所有人 |
| `--cpu-threads` | CPU 和 ONNX 使用的线程数。默认 `min(8, CPU 核数)` |
| `--weights-dir` | 权重缓存目录 |
| `--download-weights` | 下载全部权重（含 glintr100）后退出 |

内置默认值在 [`old_photo_restore/default.yaml`](old_photo_restore/default.yaml)。复制这份文件，改完后用 `--config` 传入。按人脸指定混合比例：

```yaml
blend:
  per_face:
    2: 0.55
    8: 0.60
```

编号与总览图一致（`P2` 写成 `2`）。这里的数值优先于自动降低混合比例的规则。

### 参考图，包括不同年龄的照片

把同一个人的一张或多张照片放进一个目录。参考图可以是单人照，也可以是合影，并且可以和要修复的照片相差很多年：匹配看的是身份嵌入，不是年龄，也不是表情。每张参考脸最多对应一个编号。ArcFace 余弦至少要到 `0.22`（不同年龄的同一人常常明显低于“同一天拍摄”的阈值），并且要比第二名高出一小段。只有当 `glintr100` 明显更倾向于另一个人时，才会否决这次匹配。

也可以自己指定。示例见 [`examples/refs.example.yaml`](examples/refs.example.yaml)：

```yaml
alice-2010.jpg: 2
bob.png: 8
group-photo.jpg#1: 11   # 从 0 开始，按脸的大小从大到小
```

只有参考图里有多个人时才需要 `#k`。不写的话，用这张图里最大的那张脸。

从参考图拿走的内容故意收得很窄。参考图会先按照片中这张脸的关键点对齐，然后只有眉毛、眼睛、眼镜和鼻子里的高频成分可以进入修复。嘴巴、牙齿和皮肤留在原照片上。如果某个候选结果还是改动了嘴，这块区域会用“没有看过参考图”的 CodeFormer 结果补回去。

## 输出

| 文件 | 内容 |
| --- | --- |
| `restored.png` | 原始分辨率。半尺寸图经 Real-ESRGAN x2plus，与 Lanczos 混合，再贴上修复后的脸 |
| `restored-2x.png` | 原图的两倍。半尺寸图经 Real-ESRGAN x4plus，同样混合后再贴脸 |
| `faces-overview.jpg` | 标有 `P1`…`Pn` 的总览（长边会限制在便于阅读的尺寸）。`report.txt` 里的坐标是原图像素 |
| `compare-side-by-side.jpg` | 整张图并排：原图和原始分辨率的修复结果 |
| `compare-faces.jpg` | 每张被修复的脸一对裁块：原图、修复后 |
| `compare-refs.jpg` | 仅在用了参考图时生成：原图、修复后、对齐后的参考图 |
| `report.txt` | 所用设置、每张脸的检测与身份分数、弱脸、参考图选择、耗时 |

身份分数类似这样：

```text
P3  det=0.982  blend=0.70  reason=default  identity_deblur=0.61  identity_full=0.58  identity_out=0.66
```

`identity_deblur` 是去模糊后的裁块相对原图，`identity_full` 是混合之前的 CodeFormer 结果，`identity_out` 是真正贴回去的那张脸。当完整修复结果相对“只去模糊”掉得太多、绝对分数偏低，或检测器没有把握时，混合比例会下降。这些脸列在 `weak faces:` 那一行。

## Photoshop

Photoshop 2026 已经没有“防抖 / Shake Reduction”。最接近的手动做法是：滤镜 > 锐化 > 智能锐化，移去：动感模糊，角度 0，数量 100，半径 4，减少杂色 15。那是对整张图做一次锐化，不能代替本工具按人脸做的身份检查。

## 大概要跑多久

下面是代表性运行的耗时示例。你自己这次运行的秒数写在 `report.txt` 里。

在 4 核 Linux CPU 上，一张 960×695、两张脸还带参考图搜索的合影大约 2.5 分钟。在 Apple 芯片 Mac（M 系列）上，一张单人脸测试照片（权重已经在缓存里）用 MPS 大约 55 秒：去模糊 7.8 秒，检测 1.9 秒，人脸 1.5 秒，两个尺寸的放大大约 40 秒；只出原始尺寸、改走 CPU 大约 104 秒。一张大合影主要还是花在两次放大上，会比这一张脸更久。检测在 Mac 上也走 CPU；ArcFace 可以走 CoreML。

## 排查（Apple 芯片）

- **`mps available: False`。** 装到的不是 macOS arm64 的 torch。用 arm64 的 `python3`（3.11 或 3.12）重新跑 `bash scripts/setup_mac.sh`。
- **`NotImplementedError`，或提示某个算子 “not currently implemented for the MPS device”。** 程序会在导入 PyTorch 之前设置 `PYTORCH_ENABLE_MPS_FALLBACK=1`。如果某个模型仍然失败，这个模型会改到 CPU 上继续，后面的模型仍会再试 MPS。想完全不用 MPS：`--device cpu`。
- **`MPS backend out of memory`。** 在配置里把 `deblur.tile` 和 `upscale.tile` 调小（可试 384 和 256），或直接 `--device cpu`。默认已经是半分辨率去模糊；更吃内存的是 `--deblur-scale 1`。
- **ArcFace 的 CoreML 报错。** 人脸检测不用 CoreML。ArcFace 失败时该会话会改用 CPU。人脸列表空了或不对时，用 `--device cpu` 再跑并保留日志。
- **输出全黑或出现 NaN。** 用 `--device cpu` 跑一次。PyTorch 这些步骤固定用 fp32，因为不少 MPS 内核不接受 fp16。全黑说明还有别的问题。

## 局限

- 耗时大致随人脸数量增加。每张脸是一次 512×512 的 CodeFormer（任何设备上都是一次一张）。参考图搜索对每张匹配上的脸大约是六次前向（36 组混合打分）。一张很大的合影会跑很久，不是即时预览。
- CodeFormer 仍然是一个学出来的人脸先验。保真度 `0.9` 和混合比例限制了它能偏离多远，但不是每一处身份细节都被证明保留。请看 `report.txt`。
- 人脸检测默认阈值是 `0.60`。设成 `0.40` 时，装饰性雕花或图案（分数大约 0.45 和 0.49）会被当成脸，CodeFormer 会在上面画出一张脸。五点关键点还会检查“眼睛在鼻子上方、鼻子在嘴上方”。人多、又糊、真人被漏掉时，用配置文件把阈值调低，再加 `--config`：

  ```yaml
  detect:
    threshold: 0.45
  ```
- 高光溢出、以及画面各处模糊程度不一样，正是这里不用全局反卷积核的原因。这些区域不会被完美修好。
- 另一年龄的参考图可以把眉毛、眼睛、眼镜和鼻子上的真实纹理补回来。它不会改表情，也不允许替换嘴巴。
- 不做美颜、妆容或年龄编辑，也没有手动的虹膜贴片。
- 如果某个图块内存不够，把配置里的 `deblur.tile` 和 `upscale.tile` 调小再跑（CPU 默认 384 和 256，MPS 默认 512 和 384）。已经下载的权重会留在缓存里。

## 测试

```bash
pytest                 # 单元测试，不下载模型
pytest -m smoke        # 在一张公有领域合影上做 CPU 端到端测试；会下载权重
```

冒烟测试使用 1927 年索尔维会议合影（公有领域）。不会使用或保存个人照片。

## 代码结构

阶段顺序在 `old_photo_restore/pipeline.py`。`third_party/` 里是加载已发布权重所必需的网络定义，每个文件自带许可说明。可调参数在 `default.yaml`。
