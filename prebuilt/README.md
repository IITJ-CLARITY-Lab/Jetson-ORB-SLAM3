# Prebuilt Orin binaries

`orin-jp6/` is this repository, already compiled, so a first run takes minutes
instead of the ~45 that building the CUDA kernels costs on an Orin Nano.
`run_euroc.sh` picks it up on its own — there is nothing to do here.

Built on **JetPack 6.2 (L4T R36.4.3)**, Jetson Orin Nano Super, CUDA 12.6,
TensorRT 10, from the source in this same branch.

## Will it run on my board?

`run_euroc.sh` does not guess: it executes the binary once with no arguments and
checks that it prints its usage line. If your board answers, it is used; if
anything is missing or the CPU rejects the instructions, the script says so and
compiles from source instead. Nothing silently half-works.

The two ways it can fail are worth knowing:

- **A different JetPack.** These link against OpenCV 4.8, CUDA 12 and
  `libnvinfer.so.10` as JetPack 6 ships them. JetPack 5 has none of those
  versions.
- **A Jetson that is not an Orin.** The build uses `-march=native` on a
  Cortex-A78AE, so it is fine across Orin Nano / NX / AGX and will fault on
  Xavier or the original Nano.

To ignore the prebuilt binaries deliberately, run `BUILD=1 ./run_euroc.sh`.

## What is in here

| | |
|---|---|
| `lib/libORB_SLAM3.so` | this project |
| `lib/libDBoW2.so`, `lib/libg2o.so` | `Thirdparty/`, built from the source here |
| `lib/libpangolin.so` | Pangolin 0.6 — MIT |
| `lib/libGLEW.so.2.2` | GLEW 2.2 — modified BSD |
| `lib/libboost_serialization.so.1.74.0` | Boost 1.74 — Boost Software License 1.0 |
| `bin/` | the dataset drivers from `Examples/` |
| `licenses/` | the full licence text of each third-party library above |

The three third-party libraries are here because JetPack does not ship them and
installing them needs `sudo`, which students on a shared board usually do not
have. All three are permissively licensed and redistributable, and each licence
requires its text to travel with the binary — hence `licenses/`. Everything else —
OpenCV, CUDA, cuBLAS, TensorRT, GTK, GL — comes from JetPack itself and is not
duplicated here.

`libORB_SLAM3.so` is GPLv3, like the rest of this repository, and the source
that produced it is the source next to it in this branch.

## Why they are relocatable

CMake bakes the build machine's directories into each binary as a RUNPATH. As
built, that was

```
/home/user/ORB_SLAM3_GPU/lib:/home/user/local/lib:...
```

which is actively harmful once the files move: on the machine they came from,
running one of these without `LD_LIBRARY_PATH` silently loaded the *source
tree's* `libORB_SLAM3.so` rather than the one beside it — the same
silent-substitution failure the top-level README warns about, except shipped
inside the artifact.

`fix_runpath.py` rewrote it to `$ORIGIN/../lib` for the drivers and `$ORIGIN`
for the library, so the bundle resolves against itself wherever it is unpacked.
Run one directly and it works with no environment set at all:

```bash
env -u LD_LIBRARY_PATH ldd prebuilt/orin-jp6/bin/stereo_inertial_euroc
libORB_SLAM3.so => .../prebuilt/orin-jp6/bin/../lib/libORB_SLAM3.so
```

Re-run `fix_runpath.py` after regenerating the bundle — a fresh build will have
the absolute paths back.

## Rebuilding this bundle

On an Orin with a completed build of this tree:

```bash
mkdir -p prebuilt/orin-jp6/lib prebuilt/orin-jp6/bin
cp lib/libORB_SLAM3.so Thirdparty/DBoW2/lib/libDBoW2.so \
   Thirdparty/g2o/lib/libg2o.so prebuilt/orin-jp6/lib/
cp ~/local/lib/libpangolin.so ~/local/lib/libGLEW.so.2.2 \
   ~/local/lib/libboost_serialization.so.1.74.0 prebuilt/orin-jp6/lib/
cp Examples/*/stereo_inertial_euroc Examples/*/stereo_euroc \
   Examples/*/stereo_kitti Examples/*/stereo_inertial_tum_vi \
   Examples/*/mono_inertial_euroc prebuilt/orin-jp6/bin/
strip --strip-unneeded prebuilt/orin-jp6/lib/*.so* prebuilt/orin-jp6/bin/*
python3 prebuilt/fix_runpath.py prebuilt/orin-jp6      # not optional -- see above
```

Then check nothing dangles:

```bash
LD_LIBRARY_PATH=prebuilt/orin-jp6/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64 \
  ldd prebuilt/orin-jp6/bin/stereo_inertial_euroc | grep "not found"
```
