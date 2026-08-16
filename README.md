# Jetson-ORB-SLAM3

GPU-accelerated ORB-SLAM3 for the NVIDIA Jetson Orin Nano. The ORB feature
extractor runs as CUDA kernels written to match the CPU implementation rather
than approximate it, so the accuracy of the original system is preserved; loop
closure can additionally use a CNN place-recognition front end through TensorRT.

CLARITY Lab, Indian Institute of Technology Jodhpur. GPLv3 — see `LICENSE`.

---

## Run it

On a Jetson Orin, from a clone of this branch:

```bash
./run_euroc.sh
```

That is the whole thing. The script unpacks the vocabulary, picks up the
prebuilt Orin binary shipped in `prebuilt/`, downloads EuRoC MH01 if the board
does not already have it, runs stereo-inertial SLAM, and prints the trajectory
error:

```
==> Running MH01  (GPU ORB)
    sequence: /home/user/datasets/machine_hall/machine_hall/MH_01_easy
...
==> Result
    trajectory:   /home/user/ORB_SLAM3_GPU/output/kf_MH01.txt  (312 keyframes)
    ATE RMSE:     2.14 cm   (SE(3) aligned, 298 poses)
```

Any of the eleven EuRoC sequences works:

```bash
./run_euroc.sh V101
```

A few knobs, all optional:

| | |
|---|---|
| `CPU_ORB=1 ./run_euroc.sh` | reference CPU extractor instead of the GPU one — the comparison the paper makes |
| `PIPELINE_FE=1 ./run_euroc.sh` | overlap extraction with tracking; this is what makes it real-time on an Orin |
| `BUILD=1 ./run_euroc.sh` | compile from source instead of using the prebuilt binary |
| `EUROC_DIR=/path ./run_euroc.sh` | where sequences live (default `~/datasets`) |
| `JOBS=4 ./run_euroc.sh` | compile jobs when it does build (default 2) |

With a display attached the Pangolin viewer opens; headless, it runs under
`xvfb-run` automatically.

## What to expect

**No build.** `prebuilt/orin-jp6/` holds this code already compiled for JetPack 6
on an Orin, so the first run starts in seconds. The script verifies the binary
runs on your board before trusting it, and compiles from source if it does not —
see [prebuilt/README.md](prebuilt/README.md) for exactly when that happens.
Building takes roughly 45 minutes; the CUDA kernels dominate.

MH01 is a 3-minute sequence and takes a few minutes to process. The download,
if one is needed, is about 1.5 GB per sequence and wants ~5 GB free.

Output lands in `output/`: `kf_<seq>.txt` is the keyframe trajectory in TUM
format, `f_<seq>.txt` every frame.

## Requirements

A stock **JetPack 6** Orin and nothing else — OpenCV, CUDA, cuBLAS and TensorRT
all come with it, and the three libraries JetPack lacks (Pangolin, GLEW, Boost
serialization) are in `prebuilt/orin-jp6/lib`. No `sudo` needed.

To build from source you additionally need Eigen ≥ 3.1 and Pangolin development
headers. Without `sudo`, put them under `~/local`; the script finds them there
without being told.

**Do not run `nvpmodel` or `jetson_clocks`.** They destabilise this board, and
every published number was taken at the stock profile.

## CNN loop closure

Loop closure runs on DBoW2 out of the box. The CNN path additionally needs a
CosPlace ResNet-50 ONNX model and a TensorRT engine built *on the board it will
run on* — engines are not portable:

```bash
/usr/src/tensorrt/bin/trtexec --onnx=cosplace_r50_512.onnx \
    --saveEngine=cosplace_r50_512.fp16.trt --fp16
```

Place both in the repository root. Without them you get one line on stderr
saying the engine was not found, and DBoW2 is used. Our ablation found the two
statistically indistinguishable in accuracy, so nothing is lost by leaving it
off for a first run.

## Layout

```
src/, include/     the SLAM library, including src/cuda/ (the ORB kernels)
Examples/          dataset drivers and calibration files
Thirdparty/        DBoW2, g2o, Sophus
Vocabulary/        ORB vocabulary, unpacked on first run
prebuilt/orin-jp6/ the same code already compiled for JetPack 6
run_euroc.sh       the script above
```

The KITTI, TUM and RealSense drivers all build, and EuRoC is the one wired up
end to end. TUM-VI additionally needs the per-sequence timestamp and IMU text
files; they are hundreds of megabytes of dataset-side data and were dropped from
this branch, so take them from
[upstream](https://github.com/UZ-SLAMLab/ORB_SLAM3/tree/master/Examples) if you
want that dataset.

Built on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) (Campos et al.,
T-RO 2021), with CUDA infrastructure from
[Jetson-SLAM](https://github.com/ashishkumar822/Jetson-SLAM) (Kumar et al.,
RA-L 2023). Both are GPLv3.

This branch carries the implementation only. The evaluation campaign, the
scoring scripts and the recorded results are on `main`.
