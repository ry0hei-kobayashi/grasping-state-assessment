#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====== 設定ここだけ ======
CSV_ROOT = "./tactile_raw_data"
OUT_ROOT = "./tactile_raw_plot"
# =========================


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def z_cols():
    return [f"z_t{tid:02d}" for tid in range(16)]


def plot_z16_frame(x, Z, out_png):
    """
    x: (T,) frame index
    Z: (T,16) raw tactile z (0..1)
    """
    plt.figure()
    for tid in range(16):
        plt.plot(x, Z[:, tid])
    plt.xlabel("frame_idx")
    plt.ylabel("tactile z (raw, 16 taxels)")
    plt.grid(True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ensure_dir(OUT_ROOT)

    csv_paths = sorted(glob.glob(os.path.join(CSV_ROOT, "*", "*.csv")))
    print("csv count =", len(csv_paths))

    cols = z_cols()

    for csv_path in csv_paths:
        obj = os.path.basename(os.path.dirname(csv_path))
        case = os.path.splitext(os.path.basename(csv_path))[0]

        out_dir = os.path.join(OUT_ROOT, obj)
        ensure_dir(out_dir)

        df = pd.read_csv(csv_path)

        # frame_idx がなければ行番号で代用
        if "frame_idx" in df.columns:
            x = df["frame_idx"].to_numpy(dtype=np.int32)
        else:
            x = np.arange(len(df), dtype=np.int32)

        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"{csv_path}: missing columns: {missing}")

        Z = df[cols].to_numpy(dtype=np.float32)  # (T,16)

        out_png = os.path.join(out_dir, f"{case}_z16_frame.png")
        plot_z16_frame(x, Z, out_png)

        print(f"[OK] {obj}/{case} -> {out_png}")

    print("[DONE]", OUT_ROOT)


if __name__ == "__main__":
    main()
