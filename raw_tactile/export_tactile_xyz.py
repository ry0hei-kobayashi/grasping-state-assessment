#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import numpy as np
from PIL import Image

# ====== 設定ここだけ ======
ROOT_DIR = "./graspingdata"
OUT_DIR = "./tactile_raw_data"
# =========================


def natural_key(name: str):
    m = re.match(r"(\d+)\.jpg$", name)
    return int(m.group(1)) if m else name


def list_tactile_jpgs(tactile_dir: str):
    names = [f for f in os.listdir(tactile_dir) if f.endswith(".jpg")]
    names.sort(key=natural_key)
    return [os.path.join(tactile_dir, f) for f in names]


def parse_case_name(case: str):
    # "10.1_3_1" -> width="10.1", force="3", label="1"
    parts = case.split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return "", "", ""


def read_xyz_u8(path: str):
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)  # (4,4,3)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def build_header():
    header = ["object", "case", "width", "force", "label", "frame_idx", "time_sec"]
    for tid in range(16):
        header.append(f"x_t{tid:02d}")
    for tid in range(16):
        header.append(f"y_t{tid:02d}")
    for tid in range(16):
        header.append(f"z_t{tid:02d}")
    return header


def export_one_case(obj: str, case: str):
    tactile_dir = os.path.join(ROOT_DIR, obj, case, "tactile")
    t_path = os.path.join(tactile_dir, "tactile_time_list.npy")

    if not os.path.isdir(tactile_dir) or not os.path.exists(t_path):
        return 0, None

    jpgs = list_tactile_jpgs(tactile_dir)
    if len(jpgs) == 0:
        return 0, None

    t_list = np.load(t_path)
    n = min(len(jpgs), len(t_list))
    if n == 0:
        return 0, None

    width, force, label = parse_case_name(case)

    # 出力先: OUT_DIR/object/case.csv
    out_obj_dir = os.path.join(OUT_DIR, obj)
    ensure_dir(out_obj_dir)
    out_csv = os.path.join(out_obj_dir, f"{case}.csv")

    header = build_header()

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(n):
            arr_u8 = read_xyz_u8(jpgs[i])  # (4,4,3) uint8
            arr = arr_u8.astype(np.float32) / 255.0  # 0..1
            flat = arr.reshape(16, 3)  # (taxel, xyz)

            row = [obj, case, width, force, label, i, float(t_list[i])]

            # x 16
            for tid in range(16):
                row.append(float(flat[tid, 0]))
            # y 16
            for tid in range(16):
                row.append(float(flat[tid, 1]))
            # z 16
            for tid in range(16):
                row.append(float(flat[tid, 2]))

            w.writerow(row)

    return n, out_csv


def main():
    ensure_dir(OUT_DIR)

    objects = sorted(
        [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
    )

    total_cases = 0
    total_frames = 0

    for obj in objects:
        obj_dir = os.path.join(ROOT_DIR, obj)
        cases = sorted(
            [d for d in os.listdir(obj_dir) if os.path.isdir(os.path.join(obj_dir, d))]
        )

        for case in cases:
            n, out_csv = export_one_case(obj, case)
            if n == 0:
                continue
            total_cases += 1
            total_frames += n
            print(f"[OK] {obj}/{case}: frames={n} -> {out_csv}")

    print(f"[DONE] cases={total_cases}, frames={total_frames}")
    print(f"[OUT_DIR] {OUT_DIR}")


if __name__ == "__main__":
    main()
