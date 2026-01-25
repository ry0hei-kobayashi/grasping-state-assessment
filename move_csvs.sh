#!/usr/bin/env bash
set -u  # set -e は外す（途中で止まらないように）
set -o pipefail

SRC_BASE="raw_tactile/tactile_raw_data"
DST_BASE="graspingdata_with_raw_tactile"

OBJECTS=(appbox baisui bingho cesbon cokele haitun jianjo meinad nestle nongf1 nongfu pacup1 pacup2 songsu zhijin)

if [[ ! -d "$SRC_BASE" ]]; then
  echo "[ERROR] SRC_BASE not found: $SRC_BASE" >&2
  exit 1
fi
if [[ ! -d "$DST_BASE" ]]; then
  echo "[ERROR] DST_BASE not found: $DST_BASE" >&2
  exit 1
fi

count_ok=0
count_skip_obj=0
count_skip_trial=0
count_nocsv=0
count_err=0

shopt -s nullglob

for obj in "${OBJECTS[@]}"; do
  src_dir="$SRC_BASE/$obj"
  dst_obj_base="$DST_BASE/$obj"

  if [[ ! -d "$src_dir" ]]; then
    echo "[SKIP_OBJ] source object dir not found: $src_dir"
    ((count_skip_obj++))
    continue
  fi
  if [[ ! -d "$dst_obj_base" ]]; then
    echo "[SKIP_OBJ] dataset object dir not found: $dst_obj_base"
    ((count_skip_obj++))
    continue
  fi

  csvs=("$src_dir"/*.csv)
  if (( ${#csvs[@]} == 0 )); then
    echo "[NO_CSV] $obj: no csv in $src_dir"
    ((count_nocsv++))
    continue
  fi

  for src in "${csvs[@]}"; do
    trial="$(basename "$src" .csv)"
    dst_trial_dir="$dst_obj_base/$trial"
    dst_dir="$dst_trial_dir/tactile_raw"
    dst_file="$dst_dir/$trial.csv"

    if [[ ! -d "$dst_trial_dir" ]]; then
      echo "[SKIP_TRIAL] $obj: trial dir not found: $dst_trial_dir"
      ((count_skip_trial++))
      continue
    fi

    mkdir -p "$dst_dir" || { echo "[ERR] mkdir failed: $dst_dir"; ((count_err++)); continue; }
    cp -f "$src" "$dst_file" || { echo "[ERR] copy failed: $src -> $dst_file"; ((count_err++)); continue; }

    echo "[OK] $obj: $src -> $dst_file"
    ((count_ok++))
  done
done

echo "----"
echo "OK         : $count_ok"
echo "SKIP_OBJ   : $count_skip_obj"
echo "SKIP_TRIAL : $count_skip_trial"
echo "NO_CSV     : $count_nocsv"
echo "ERR        : $count_err"


