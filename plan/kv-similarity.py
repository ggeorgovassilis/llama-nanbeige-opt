#!/usr/bin/env python3
"""Parse a kv-probe dump and compare pass-1 vs pass-2 K/V for the Nanbeige looped model.

Reports cosine similarity and relative L2 error between logical layer i (pass 1) and
layer i + n_phys (pass 2) for each physical layer, averaged over the dumped positions.
"""

import struct
import sys

import numpy as np

MAGIC = 0x4B565052
VERSION = 1

# Nanbeige 4.2: 22 physical layers, 2 loops.
N_PHYS = 22
N_LOOPS = 2


def dequant_q8_0(raw: bytes, nembd: int) -> np.ndarray:
    # ggml block_q8_0: { ggml_half d (2 bytes fp16); int8_t qs[32] } = 34 bytes per 32 values
    nb = nembd // 32
    arr = np.frombuffer(raw, dtype=np.uint8)
    assert arr.size == nb * 34, f"row size {arr.size} != {nb}*34 for nembd {nembd}"
    blk = arr.reshape(nb, 34)
    d = blk[:, :2].copy().view(np.float16).astype(np.float32).reshape(nb)
    q = blk[:, 2:].astype(np.int8).astype(np.float32)
    return (q * d[:, None]).reshape(-1).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = max(np.linalg.norm(a), np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b) / denom)


def main(path: str) -> None:
    with open(path, "rb") as f:
        data = f.read()

    off = 0

    def read(fmt: str):
        nonlocal off
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, data, off)
        off += size
        return val[0] if len(val) == 1 else val

    magic = read("<I")
    version = read("<I")
    assert magic == MAGIC, f"bad magic {magic:#x}"
    assert version == VERSION, f"bad version {version}"

    n_layer = read("<I")
    n_pos = read("<I")

    print(f"n_layer={n_layer} n_pos={n_pos}")

    layers = []  # (il, k_type, k_nembd, k_row, v_type, v_nembd, v_row)
    for _ in range(n_layer):
        il = read("<i")
        k_type = read("<I")
        k_nembd = read("<I")
        k_row = read("<I")
        v_type = read("<I")
        v_nembd = read("<I")
        v_row = read("<I")
        layers.append((il, k_type, k_nembd, k_row, v_type, v_nembd, v_row))

    # map logical layer -> parsed row data
    k_rows = {}
    v_rows = {}
    for (il, k_type, k_nembd, k_row, v_type, v_nembd, v_row) in layers:
        ks = []
        vs = []
        for _ in range(n_pos):
            kraw = data[off:off + k_row]
            off += k_row
            vraw = data[off:off + v_row]
            off += v_row
            ks.append(dequant_q8_0(kraw, k_nembd))
            vs.append(dequant_q8_0(vraw, v_nembd))
        k_rows[il] = ks
        v_rows[il] = vs

    print(f"physical layers: {N_PHYS}, loops: {N_LOOPS}")
    print()
    print(f"{'phys':>4} | {'K cos (mean/min)':>22} | {'K relL2 (mean)':>16} | {'V cos (mean/min)':>22} | {'V relL2 (mean)':>16}")
    print("-" * 96)

    k_cos_all = []
    v_cos_all = []
    for i in range(N_PHYS):
        j = i + N_PHYS
        if i not in k_rows or j not in k_rows:
            print(f"layer {i} or {j} missing")
            continue
        kc = [cosine(k_rows[i][p], k_rows[j][p]) for p in range(n_pos)]
        vc = [cosine(v_rows[i][p], v_rows[j][p]) for p in range(n_pos)]
        kl = [rel_l2(k_rows[i][p], k_rows[j][p]) for p in range(n_pos)]
        vl = [rel_l2(v_rows[i][p], v_rows[j][p]) for p in range(n_pos)]
        k_cos_all.extend(kc)
        v_cos_all.extend(vc)
        print(
            f"{i:>4} | {np.mean(kc):>10.4f} / {np.min(kc):>10.4f} | {np.mean(kl):>16.4f} "
            f"| {np.mean(vc):>10.4f} / {np.min(vc):>10.4f} | {np.mean(vl):>16.4f}"
        )

    print("-" * 96)
    print(f"overall K cosine mean/min = {np.mean(k_cos_all):.4f} / {np.min(k_cos_all):.4f}")
    print(f"overall V cosine mean/min = {np.mean(v_cos_all):.4f} / {np.min(v_cos_all):.4f}")

    # per-position breakdown for a few representative physical layers
    print()
    print("per-position K cosine (layer 0, 5, 10, 21):")
    hdr = "".join(f"{p:>7}" for p in range(n_pos))
    print(f"{'layer':>6} |{hdr}")
    for i in (0, 5, 10, 21):
        row = " ".join(f"{cosine(k_rows[i][p], k_rows[i + N_PHYS][p]):>7.3f}" for p in range(n_pos))
        print(f"{i:>6} | {row}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} DUMPFILE", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
