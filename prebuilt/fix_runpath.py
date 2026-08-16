#!/usr/bin/env python3
"""Rewrite the RUNPATH of the shipped binaries to be self-contained.

CMake bakes the build machine's library directories into every binary it links.
For this bundle that meant a RUNPATH of

    /home/user/ORB_SLAM3_GPU/lib:/home/user/local/lib:...

which is worse than useless once the files move: on the machine they were built
on, running one of these binaries without LD_LIBRARY_PATH silently loads the
*source tree's* libORB_SLAM3.so instead of the one sitting next to it. That is
the same silent-substitution failure the top-level README warns about, except
baked into the artifact.

This rewrites it to $ORIGIN-relative, so the bundle resolves against itself
wherever it is unpacked and cannot pick up a stranger's build.

patchelf is the usual tool for this and is not installable on a Jetson without
sudo, so this does the edit directly: the new string is written over the old one
inside .dynstr, which is safe because it is strictly shorter and the entry is
NUL-terminated. No section is moved, resized, or reordered.

    python3 fix_runpath.py orin-jp6
"""
import struct
import sys
from pathlib import Path

DT_RPATH, DT_RUNPATH = 15, 29


def sections(buf):
    """Yield (name, offset, size) for each section header."""
    e_shoff, = struct.unpack_from('<Q', buf, 0x28)
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from('<HHH', buf, 0x3A)
    hdr = lambda i: struct.unpack_from('<IIQQQQ', buf, e_shoff + i * e_shentsize)
    _, _, _, _, str_off, _ = hdr(e_shstrndx)
    for i in range(e_shnum):
        sh_name, _, _, _, sh_offset, sh_size = hdr(i)
        end = buf.index(b'\0', str_off + sh_name)
        yield buf[str_off + sh_name:end].decode(), sh_offset, sh_size


def patch(path, new):
    buf = bytearray(path.read_bytes())
    if buf[:4] != b'\x7fELF':
        return 'not an ELF'
    sec = {n: (o, s) for n, o, s in sections(buf)}
    if '.dynamic' not in sec or '.dynstr' not in sec:
        return 'no dynamic section'
    dyn_off, dyn_size = sec['.dynamic']
    str_off, _ = sec['.dynstr']

    for p in range(dyn_off, dyn_off + dyn_size, 16):
        tag, val = struct.unpack_from('<qQ', buf, p)
        if tag == 0:                                  # DT_NULL, end of table
            break
        if tag not in (DT_RPATH, DT_RUNPATH):
            continue
        at = str_off + val
        old = buf[at:buf.index(b'\0', at)]
        enc = new.encode()
        if len(enc) > len(old):
            return 'new RUNPATH is longer than the old one (%d > %d)' % (len(enc), len(old))
        buf[at:at + len(old) + 1] = enc + b'\0' * (len(old) - len(enc) + 1)
        path.write_bytes(buf)
        return 'was %s' % old.decode()
    return 'no RUNPATH set'


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else 'orin-jp6')
    # Binaries live one level below the libraries they need; the libraries sit
    # beside each other. CUDA stays absolute -- it belongs to the platform.
    for sub, rpath in (('bin', '$ORIGIN/../lib:/usr/local/cuda/lib64'),
                       ('lib', '$ORIGIN:/usr/local/cuda/lib64')):
        for f in sorted((root / sub).iterdir()):
            if f.is_file():
                print('%-40s %s' % (f, patch(f, rpath)))


if __name__ == '__main__':
    main()
