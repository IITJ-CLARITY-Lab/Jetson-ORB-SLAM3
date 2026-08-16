#!/usr/bin/env bash
#
# Jetson-ORB-SLAM3 -- one-command EuRoC demo on a Jetson Orin.
#
#   ./run_euroc.sh              # MH01
#   ./run_euroc.sh V101         # MH01..MH05, V101..V103, V201..V203
#
# Uses the prebuilt Orin binary in prebuilt/ (or builds, if that will not run
# here), fetches the sequence if it is not on the board, runs stereo-inertial
# SLAM, and prints the trajectory error against ground truth. Nothing else to
# install, nothing else to configure.
#
# Optional knobs:
#   CPU_ORB=1        run the reference CPU feature extractor instead of the GPU one
#   PIPELINE_FE=1    overlap feature extraction with tracking (faster, real-time)
#   BUILD=1          compile from source, ignoring the prebuilt binary
#   EUROC_DIR=path   where sequences live / are downloaded to  (default ~/datasets)
#   JOBS=n           compile jobs for the build                 (default 2)
#
# Copyright (C) 2026 CLARITY Lab, Indian Institute of Technology Jodhpur.
# GNU General Public License v3 or later; see LICENSE.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEQ="${1:-MH01}"
DATA="${EUROC_DIR:-$HOME/datasets}"
BIN="$ROOT/Examples/Stereo-Inertial/stereo_inertial_euroc"

case "$SEQ" in
  MH01) NAMES="MH_01_easy MH_01 MH01";           BUNDLE=machine_hall ;;
  MH02) NAMES="MH_02_easy MH_02 MH02";           BUNDLE=machine_hall ;;
  MH03) NAMES="MH_03_medium MH_03 MH03";         BUNDLE=machine_hall ;;
  MH04) NAMES="MH_04_difficult MH_04 MH04";      BUNDLE=machine_hall ;;
  MH05) NAMES="MH_05_difficult MH_05 MH05";      BUNDLE=machine_hall ;;
  V101) NAMES="V1_01_easy V1_01 V101";           BUNDLE=vicon_room1 ;;
  V102) NAMES="V1_02_medium V1_02 V102";         BUNDLE=vicon_room1 ;;
  V103) NAMES="V1_03_difficult V1_03 V103";      BUNDLE=vicon_room1 ;;
  V201) NAMES="V2_01_easy V2_01 V201";           BUNDLE=vicon_room2 ;;
  V202) NAMES="V2_02_medium V2_02 V202";         BUNDLE=vicon_room2 ;;
  V203) NAMES="V2_03_difficult V2_03 V203";      BUNDLE=vicon_room2 ;;
  *) echo "unknown sequence '$SEQ' -- use MH01..MH05, V101..V103, V201..V203" >&2; exit 1 ;;
esac
CANON="${NAMES%% *}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- vocabulary
if [ ! -f "$ROOT/Vocabulary/ORBvoc.txt" ]; then
  say "Unpacking ORB vocabulary (once, ~30 s)"
  tar -xzf "$ROOT/Vocabulary/ORBvoc.txt.tar.gz" -C "$ROOT/Vocabulary"
fi

# ------------------------------------------------------------------ prebuilt
# A build for JetPack 6 Orin ships in the branch so nobody has to spend 45
# minutes on CUDA before seeing anything. It is used only if it actually runs
# here: a different JetPack, or a Jetson that is not an Orin, fails the check
# below and falls through to compiling from source rather than dying mid-run.
LIBS="$ROOT/lib:$HOME/local/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$ROOT/Thirdparty/DBoW2/lib:$ROOT/Thirdparty/g2o/lib"
PRE="$ROOT/prebuilt/orin-jp6"
if [ ! -x "$BIN" ] && [ -z "${BUILD:-}" ] && [ -x "$PRE/bin/stereo_inertial_euroc" ]; then
  PRELIBS="$PRE/lib:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64"
  # Run it with no arguments: a working binary prints its usage line. Capture
  # first rather than piping -- the driver exits 1 on a usage error, and under
  # `pipefail` that sinks the whole pipeline even when grep matches, which reads
  # as "this board cannot run it" and silently costs 45 minutes of compiling.
  PROBE=$(LD_LIBRARY_PATH="$PRELIBS" "$PRE/bin/stereo_inertial_euroc" 2>&1 || true)
  if printf '%s\n' "$PROBE" | grep -q '^Usage'; then
    say "Using the prebuilt Orin binary  (BUILD=1 compiles from source instead)"
    BIN="$PRE/bin/stereo_inertial_euroc"
    LIBS="$PRELIBS"
  else
    say "The prebuilt binary does not run on this board -- building from source"
  fi
fi

# --------------------------------------------------------------------- build
if [ ! -x "$BIN" ]; then
  say "Building -- first time only, about 45 minutes on an Orin Nano"
  # Non-interactive shells do not read ~/.bashrc, so a user-space Eigen/Pangolin
  # under ~/local is invisible to find_package unless we say so explicitly.
  if [ -d "$HOME/local/include" ]; then
    export CMAKE_PREFIX_PATH="$HOME/local${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    export CPLUS_INCLUDE_PATH="$HOME/local/include/eigen3:$HOME/local/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  fi
  for t in DBoW2 g2o Sophus; do
    cmake -S "$ROOT/Thirdparty/$t" -B "$ROOT/Thirdparty/$t/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$ROOT/Thirdparty/$t/build" -j "$(nproc)"
  done
  cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
  # -j2 by default: the CUDA kernels are memory-hungry and -j$(nproc) OOMs an 8 GB Orin.
  cmake --build "$ROOT/build" -j "${JOBS:-2}"
fi
[ -x "$BIN" ] || { echo "build finished but $BIN is missing" >&2; exit 1; }

# ------------------------------------------------------------------- dataset
SEQPATH=""
for n in $NAMES; do
  for d in "$DATA/$n" "$DATA"/*/"$n" "$DATA"/*/*/"$n"; do
    if [ -d "$d/mav0/cam0/data" ]; then SEQPATH="$d"; break 2; fi
  done
done

if [ -z "$SEQPATH" ]; then
  say "$SEQ is not under $DATA -- downloading (~1.5 GB, needs ~5 GB free)"
  SEQPATH="$DATA/$CANON"
  mkdir -p "$SEQPATH"
  # Check up front: the sequence arrives as one deflated stream that has to be
  # written out whole before it can be unpacked, so running out of room happens
  # after the download, not before it.
  FREE=$(df -Pk "$SEQPATH" | awk 'NR==2 {print int($4/1048576)}')
  [ "$FREE" -ge 5 ] || { echo "only ${FREE} GB free on $(df -Ph "$SEQPATH" | awk 'NR==2{print $6}') -- need 5" >&2; exit 1; }
  BUNDLE="$BUNDLE" CANON="$CANON" DEST="$SEQPATH" python3 - <<'PY'
import os, struct, sys, urllib.request, zipfile, zlib

# The dataset's original robotics.ethz.ch home is gone. This is the ETH Research
# Collection copy: three multi-gigabyte bundles, each holding one nested .zip per
# sequence. We range-GET only the nested zip we need instead of the whole bundle.
BITSTREAM = {'machine_hall': '7b2419c1-62b5-4714-b7f8-485e5fe3e5fe',
             'vicon_room1':  '02ecda9a-298f-498b-970c-b7c44334d880',
             'vicon_room2':  'ea12bc01-3677-4b4c-853d-87c7870b8c44'}
URL = ('https://www.research-collection.ethz.ch/server/api/core/bitstreams/%s/content'
       % BITSTREAM[os.environ['BUNDLE']])
# A non-browser User-Agent is answered with a "scraping detected" HTML page, not a 403.
UA = ('Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) '
      'Chrome/126.0 Safari/537.36')
CANON, DEST = os.environ['CANON'], os.environ['DEST']
TMP = DEST + '.inner.zip'


def rng(a, b=''):
    req = urllib.request.Request(URL, headers={'User-Agent': UA, 'Range': 'bytes=%s-%s' % (a, b)})
    return urllib.request.urlopen(req, timeout=120)


size = int(rng(0, 0).headers['Content-Range'].split('/')[1])
tail = rng(size - 66000, size - 1).read()

i = tail.rfind(b'PK\x06\x06')                       # ZIP64 end-of-central-directory
if i >= 0:
    cdsz, cdoff = struct.unpack('<QQ', tail[i + 40:i + 56])
else:
    i = tail.rfind(b'PK\x05\x06')
    cdsz, cdoff = struct.unpack('<II', tail[i + 12:i + 20])
cd = rng(cdoff, cdoff + cdsz - 1).read()

entry, p = None, 0
while p < len(cd) and cd[p:p + 4] == b'PK\x01\x02':
    csz, usz = struct.unpack('<II', cd[p + 20:p + 28])
    nl, el, cl = struct.unpack('<HHH', cd[p + 28:p + 34])
    lho = struct.unpack('<I', cd[p + 42:p + 46])[0]
    name = cd[p + 46:p + 46 + nl].decode('utf8', 'replace')
    ex = cd[p + 46 + nl:p + 46 + nl + el]
    if 0xffffffff in (csz, usz, lho):               # ZIP64 extra field
        q = 0
        while q + 4 <= len(ex):
            hid, hsz = struct.unpack('<HH', ex[q:q + 4])
            blk, k = ex[q + 4:q + 4 + hsz], 0
            if hid == 1:
                if usz == 0xffffffff: usz = struct.unpack('<Q', blk[k:k + 8])[0]; k += 8
                if csz == 0xffffffff: csz = struct.unpack('<Q', blk[k:k + 8])[0]; k += 8
                if lho == 0xffffffff: lho = struct.unpack('<Q', blk[k:k + 8])[0]
            q += 4 + hsz
    if name.endswith('/%s/%s.zip' % (CANON, CANON)):
        entry = (lho, csz)
        break
    p += 46 + nl + el + cl
if entry is None:
    sys.exit('%s.zip not found inside the %s bundle' % (CANON, os.environ['BUNDLE']))

lho, csz = entry
hdr = rng(lho, lho + 29).read()                     # local header: name/extra lengths
nl, el = struct.unpack('<HH', hdr[26:30])
start = lho + 30 + nl + el

# The nested zip is deflated, so it cannot be indexed into -- inflate it whole,
# then extract only the four directories we actually use. The temp file is 1.5 GB,
# so it goes away on the way out however this ends.
KEEP = ('cam0/', 'cam1/', 'imu0/', 'state_groundtruth_estimate0/')
try:
    r = rng(start, start + csz - 1)
    dec, got = zlib.decompressobj(-15), 0
    with open(TMP, 'wb') as out:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(dec.decompress(chunk))
            got += len(chunk)
            sys.stdout.write('\r    %5.1f%%  %.2f GB' % (100.0 * got / csz, got / 1e9))
            sys.stdout.flush()
        out.write(dec.flush())
    print()

    with zipfile.ZipFile(TMP) as z:
        for m in z.namelist():
            j = m.find('mav0/')
            if j < 0 or m.endswith('/') or not m[j + 5:].startswith(KEEP):
                continue
            tgt = os.path.join(DEST, m[j:])
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            with z.open(m) as src, open(tgt, 'wb') as dst:
                dst.write(src.read())
finally:
    if os.path.exists(TMP):
        os.remove(TMP)
print('    extracted to', DEST)
PY
  # Ground truth as well as images: a half-extracted sequence must not look done
  # to the next run.
  [ -s "$SEQPATH/mav0/state_groundtruth_estimate0/data.csv" ] && [ -d "$SEQPATH/mav0/cam0/data" ] \
    || { echo "download did not fully populate $SEQPATH/mav0" >&2; exit 1; }
fi

# ----------------------------------------------------------------------- run
# Each build must see only its own libORB_SLAM3.so. Inheriting an outer
# LD_LIBRARY_PATH is how you silently end up running someone else's library, so
# this is set outright rather than prepended.
export LD_LIBRARY_PATH="$LIBS"
unset ADAPTIVE_LC DYNAMIC_MASK_ENGINE       # unfinished experiments, off by default

TAG="$SEQ"
mkdir -p "$ROOT/output"
MODE="GPU ORB"; [ -n "${CPU_ORB:-}" ] && MODE="reference CPU ORB"
[ -n "${PIPELINE_FE:-}" ] && MODE="$MODE, pipelined front end"
say "Running $SEQ  ($MODE)"
echo "    sequence: $SEQPATH"

cd "$ROOT"
ARGS=(Vocabulary/ORBvoc.txt Examples/Stereo-Inertial/EuRoC.yaml
      "$SEQPATH" "Examples/Stereo-Inertial/EuRoC_TimeStamps/$SEQ.txt" "$TAG")
# The viewer is always constructed, so a headless board needs a virtual display
# or Pangolin aborts before the first frame.
if [ -n "${DISPLAY:-}" ]; then
  "$BIN" "${ARGS[@]}"
else
  command -v xvfb-run >/dev/null \
    || { echo "no DISPLAY and no xvfb-run -- install xvfb, or ssh with -X" >&2; exit 1; }
  xvfb-run -a "$BIN" "${ARGS[@]}"
fi

for f in "kf_$TAG.txt" "f_$TAG.txt"; do
  [ -f "$ROOT/$f" ] && mv "$ROOT/$f" "$ROOT/output/$f"
done
KF="$ROOT/output/kf_$TAG.txt"
[ -f "$KF" ] || { echo "no trajectory written -- see the output above" >&2; exit 1; }

# --------------------------------------------------------------------- score
GT="$SEQPATH/mav0/state_groundtruth_estimate0/data.csv"
say "Result"
echo "    trajectory:   $KF  ($(wc -l < "$KF" | tr -d ' ') keyframes)"
if [ -f "$GT" ] && python3 -c 'import numpy' 2>/dev/null; then
  python3 - "$KF" "$GT" <<'PY'
import sys
import numpy as np


def load(path, sep, cols, skip_hash=True):
    out = {}
    for line in open(path):
        if skip_hash and line.startswith('#'):
            continue
        v = line.replace(sep, ' ').split()
        if len(v) > max(cols):
            out[float(v[0])] = np.array([float(v[c]) for c in cols])
    return out


est = load(sys.argv[1], ' ', (1, 2, 3))
# ORB-SLAM3 writes EuRoC stamps in nanoseconds, but seconds in some builds.
scale = 1e9 if est and next(iter(est)) > 1e12 else 1.0
est = {t / scale: p for t, p in est.items()}
gt = {t / 1e9: p for t, p in load(sys.argv[2], ',', (1, 2, 3)).items()}

ts = np.array(sorted(gt))
E, G = [], []
for t in sorted(est):                      # nearest ground-truth sample within 20 ms
    i = np.searchsorted(ts, t)
    best = min(((abs(ts[j] - t), j) for j in (i - 1, i) if 0 <= j < len(ts)),
               default=None)
    if best and best[0] < 0.02:
        E.append(est[t]); G.append(gt[ts[best[1]]])
if len(E) < 10:
    sys.exit('    too few matches against ground truth to score')

E, G = np.array(E), np.array(G)
# Umeyama with the scale fixed at 1: in inertial mode metric scale is observable,
# so fitting it away would flatter the result.
ec, gc = E - E.mean(0), G - G.mean(0)
U, S, Vt = np.linalg.svd(ec.T @ gc)
D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
R = Vt.T @ D @ U.T
A = E @ R.T + (G.mean(0) - R @ E.mean(0))
print('    ATE RMSE:     %.2f cm   (SE(3) aligned, %d poses)'
      % (np.sqrt(np.mean(np.sum((A - G) ** 2, 1))) * 100, len(E)))
PY
fi
echo
