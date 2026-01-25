import os, glob
import pandas as pd
import numpy as np

CSV_PATH = "./graspingdata_with_raw_tactile/pacup2/11.3_24_2/tactile_raw/11.3_24_2.csv"  # ここをCSVファイルに
IMG_DIR = "./graspingdata/pacup2/11.3_24_2/visual"  # ここは貼ってくれたディレクトリ

df = pd.read_csv(CSV_PATH)

# 画像側のframe_idx集合（例: 12.jpg -> 12）
img_ids = set()
for p in glob.glob(os.path.join(IMG_DIR, "*.jpg")):
    base = os.path.basename(p)
    try:
        img_ids.add(int(os.path.splitext(base)[0]))
    except:
        pass

csv_ids = set(df["frame_idx"].astype(int).tolist())

missing_imgs = sorted(csv_ids - img_ids)
extra_imgs = sorted(img_ids - csv_ids)

print("CSV frames:", (min(csv_ids), max(csv_ids)), "count=", len(csv_ids))
print("IMG frames:", (min(img_ids), max(img_ids)), "count=", len(img_ids))
print(
    "Missing images for frame_idx (first 50):",
    missing_imgs[:50],
    " ... total=",
    len(missing_imgs),
)
print(
    "Images not referenced by CSV (first 50):",
    extra_imgs[:50],
    " ... total=",
    len(extra_imgs),
)
