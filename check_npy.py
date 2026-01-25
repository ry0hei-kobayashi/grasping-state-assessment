import numpy as np
from pathlib import Path

base_path = "./graspingdata/pacup2/11.3_24_2/"
t_path = Path(base_path + "tactile/tactile_time_list.npy")
v_path = Path(base_path + "visual/visual_time_list.npy")

t = np.load(t_path, allow_pickle=True)
v = np.load(v_path, allow_pickle=True)


def summarize(name, a):
    a = np.asarray(a)
    print(f"\n=== {name} ===")
    print("path:", (t_path if name.startswith("tactile") else v_path))
    print("shape:", a.shape, "dtype:", a.dtype)
    # flatten for easier stats
    af = a.reshape(-1)
    print("len:", len(af))
    print("head:", af[:10])
    print("tail:", af[-10:])
    # numeric stats if possible
    try:
        af_num = af.astype(np.float64)
        print("min/max:", float(np.min(af_num)), float(np.max(af_num)))
        if len(af_num) >= 2:
            d = np.diff(af_num)
            print(
                "diff stats (min/med/max):",
                float(np.min(d)),
                float(np.median(d)),
                float(np.max(d)),
            )
            print("monotonic nondecreasing:", bool(np.all(d >= 0)))
    except Exception as e:
        print("numeric stats: (skip)", e)


summarize("tactile_time_list", t)
summarize("visual_time_list", v)

# ---- alignment check (nearest visual timestamp for each tactile timestamp) ----
# assumes both are numeric timestamps in same unit
try:
    tf = np.asarray(t).reshape(-1).astype(np.float64)
    vf = np.asarray(v).reshape(-1).astype(np.float64)

    # sort visual just in case
    order = np.argsort(vf)
    vf_sorted = vf[order]

    idx = np.searchsorted(vf_sorted, tf, side="left")
    idx0 = np.clip(idx - 1, 0, len(vf_sorted) - 1)
    idx1 = np.clip(idx, 0, len(vf_sorted) - 1)

    # choose closer neighbor
    d0 = np.abs(tf - vf_sorted[idx0])
    d1 = np.abs(tf - vf_sorted[idx1])
    best = np.where(d1 < d0, idx1, idx0)
    best_v = vf_sorted[best]
    dt = tf - best_v

    print("\n=== Nearest alignment tactile -> visual ===")
    print("tactile_len:", len(tf), "visual_len:", len(vf))
    print(
        "abs(dt) stats (min/med/max):",
        float(np.min(np.abs(dt))),
        float(np.median(np.abs(dt))),
        float(np.max(np.abs(dt))),
    )
    print("first 20 mappings: tactile_i -> visual_i (sorted-visual-index), dt")
    for i in range(min(20, len(tf))):
        print(f"{i:3d} -> {int(best[i]):3d}, dt={dt[i]: .6f}")
except Exception as e:
    print("\n(alignment check skipped):", e)
