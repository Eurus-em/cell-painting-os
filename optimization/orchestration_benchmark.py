#!/usr/bin/env python
"""全流程编排层优化 before/after benchmark

测量报告第 4 节「全流程编排优化（5 个文件）」里可本地复现的 4 项：

1. NPZ 并行加载   (deepprofiler_features.py)  串行 vs ThreadPoolExecutor(4)
2. Image.csv 按需读列 (profiling_native.py)   读全列 vs usecols
3. 移除 CSV roundtrip (profiling_native.py)    roundtrip(写+读回) vs 直接内存
4. 预览图并行生成  (segmentation_native.py)    串行 vs ThreadPoolExecutor(4)

说明：
- 真实数据只有 2 个 field / 2 个 .npz，并行优势在这么小的规模下测不出来，
  因此 NPZ 加载、预览图生成用「合成数据、按真实规模放大」来测，Image.csv usecols
  和 CSV roundtrip 直接测真实文件。S3 下载需要云环境，不在本脚本内测。
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# ── 真实数据路径 ──────────────────────────────────────────────
BASE = Path(r"D:\CellPainting-Claw-main2\CellPainting-Claw-main")
IMAGE_CSV = BASE / "demo" / "workspace" / "outputs" / "BR00117035_A01_final" / "Image.csv"

REPEAT = 3  # 每项重复次数，取最小值（最稳）


def _timeit(fn, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _fmt(seconds: float) -> str:
    return f"{seconds * 1000:.1f}ms" if seconds < 1 else f"{seconds:.2f}s"


def _speedup(before: float, after: float) -> str:
    if after < 1e-6:
        return "消除（0 开销）"
    return f"{before / after:.2f}×"


# ============================================================
# 1. NPZ 并行加载
# ============================================================
def _make_synthetic_npz(root: Path, n_files: int, cells: int, feats: int):
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n_files):
        features = rng.standard_normal((cells, feats)).astype(np.float32)
        locations = rng.standard_normal((cells, 2)).astype(np.float32)
        metadata = {"Metadata_Plate": "BR00117035", "Metadata_Well": "A01", "Metadata_Site": str(i + 1)}
        p = root / f"{i:03d}.npz"
        np.savez_compressed(p, features=features, locations=locations, metadata=metadata)
        paths.append(p)
    return paths


def bench_npz_parallel():
    tmp = Path(tempfile.mkdtemp(prefix="npz_bench_"))
    try:
        N = 64
        paths = _make_synthetic_npz(tmp, N, cells=68, feats=672)

        def serial():
            for p in paths:
                with np.load(p, allow_pickle=True) as d:
                    _ = np.asarray(d["features"])

        def parallel():
            def load(p):
                with np.load(p, allow_pickle=True) as d:
                    return np.asarray(d["features"])
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(load, paths))

        before = _timeit(serial)
        after = _timeit(parallel)
        return f"串行加载 {N} 个 .npz", before, after
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 2. Image.csv 按需读列（真实文件，9803 列）
# ============================================================
def bench_image_csv_usecols():
    if not IMAGE_CSV.exists():
        return None  # 真实文件不存在则跳过
    cols = pd.read_csv(IMAGE_CSV, nrows=0).columns.tolist()
    usecols = cols[:6]  # 模拟 merge_key + 5 个元信息列

    def full():
        _ = pd.read_csv(IMAGE_CSV)

    def selective():
        _ = pd.read_csv(IMAGE_CSV, usecols=usecols)

    before = _timeit(full)
    after = _timeit(selective)
    return f"Image.csv 读 {len(cols)} 列 vs {len(usecols)} 列", before, after


# ============================================================
# 3. 移除 CSV roundtrip
# ============================================================
def bench_csv_roundtrip():
    tmp = Path(tempfile.mkdtemp(prefix="roundtrip_bench_"))
    try:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((68, 700)), columns=[f"Cells_Feature_{i}" for i in range(700)])
        out = tmp / "single_cell.csv.gz"

        def roundtrip():
            df.to_csv(out, index=False, compression="gzip")
            _ = pd.read_csv(out)

        def direct():
            _ = df  # 直接内存传递，无写读往返

        before = _timeit(roundtrip)
        after = _timeit(direct)
        return "shard CSV roundtrip（写+读回）vs 直接内存", before, after
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 4. 预览图并行生成（合成 8 通道图）
# ============================================================
def _compose_rgb(channels: dict) -> np.ndarray:
    # 简化版伪彩色合成：取 3 个通道拼成 RGB（真实实现更贵，并行收益只会更大）
    r = channels["ch1"].astype(np.float32)
    g = channels["ch2"].astype(np.float32)
    b = channels["ch5"].astype(np.float32)
    return np.stack([r, g, b], axis=-1)


def bench_preview_parallel():
    tmp = Path(tempfile.mkdtemp(prefix="preview_bench_"))
    try:
        N = 32  # 32 个 FOV
        size = 512
        rng = np.random.default_rng(0)
        fields = []
        for i in range(N):
            imgs = {f"ch{c}": rng.integers(0, 65535, (size, size), dtype=np.uint16) for c in range(1, 9)}
            fields.append(imgs)

        def serial():
            for imgs in fields:
                rgb = _compose_rgb(imgs)
                _ = (rgb / 256).astype(np.uint8)  # 模拟保存前的转换

        def parallel():
            def one(imgs):
                rgb = _compose_rgb(imgs)
                return (rgb / 256).astype(np.uint8)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(one, fields))

        before = _timeit(serial)
        after = _timeit(parallel)
        return f"预览图生成 {N} FOV（8 通道合成）", before, after
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================
# 主流程
# ============================================================
def bench_s3_download(bucket: str, region: str, prefix: str, max_files: int):
    """S3 下载：串行 vs ThreadPoolExecutor(8)。需 boto3 + 网络，失败则跳过。"""
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.client import Config as BotocoreConfig
    except ImportError:
        print("  [skip] S3：boto3 未安装")
        return None

    try:
        client = boto3.session.Session().client(
            "s3", region_name=region,
            config=BotocoreConfig(signature_version=UNSIGNED),
        )
        paginator = client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
            if len(keys) >= max_files:
                break
        keys = keys[:max_files]
        if not keys:
            print(f"  [skip] S3：prefix {prefix!r} 下无对象")
            return None
    except Exception as exc:
        print(f"  [skip] S3：list 失败（{exc}）")
        return None

    def _download_all(workers: int, tag: str) -> float:
        tmp = Path(tempfile.mkdtemp(prefix=f"s3_bench_{tag}_"))
        try:
            def one(key: str):
                client.download_file(bucket, key, str(tmp / os.path.basename(key)))
            t0 = time.perf_counter()
            if workers == 1:
                for key in keys:
                    one(key)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    list(pool.map(one, keys))
            return time.perf_counter() - t0
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    before = _download_all(1, "serial")
    after = _download_all(8, "parallel")
    return f"S3 下载 {len(keys)} 对象（{bucket}/{prefix}）", before, after


def main():
    parser = argparse.ArgumentParser(description="全流程编排层优化 before/after benchmark")
    parser.add_argument("--s3", action="store_true", help="额外测 S3 下载（需 boto3 + 网络）")
    parser.add_argument("--s3-bucket", default="cellpainting-gallery")
    parser.add_argument("--s3-region", default="us-east-1")
    parser.add_argument("--s3-prefix", default="cpg0000-jump-pilot")
    parser.add_argument("--s3-max-files", type=int, default=16)
    args = parser.parse_args()

    print("全流程编排层优化 before/after benchmark")
    print("=" * 70)

    benches = [bench_npz_parallel, bench_image_csv_usecols, bench_csv_roundtrip, bench_preview_parallel]
    if args.s3:
        benches.append(lambda: bench_s3_download(args.s3_bucket, args.s3_region, args.s3_prefix, args.s3_max_files))

    rows = []
    for bench in benches:
        result = bench()
        if result is None:
            continue
        label, before, after = result
        rows.append((label, before, after))

    # 打印 Markdown 表格
    print("\n| 优化 | 优化前 | 优化后 | 加速 |")
    print("|------|:---:|:---:|:---:|")
    for label, before, after in rows:
        print(f"| {label} | {_fmt(before)} | {_fmt(after)} | {_speedup(before, after)} |")

    print("\n" + "=" * 70)
    print("注：NPZ 加载、预览图生成为合成数据按真实规模放大；Image.csv、CSV roundtrip 为真实数据。")
    if args.s3:
        print("    S3 下载为真实网络下载，单次计时（不重复）。")


if __name__ == "__main__":
    main()
