import subprocess, time, sys, re
from pathlib import Path

OUT = Path("D:/CellPainting-Claw-main/CellProfiler-main/run_timed")
if OUT.exists():
    import shutil; shutil.rmtree(OUT)
OUT.mkdir(parents=True)

cmd = [
    "D:/MINICONDA/envs/cellpainting-claw/Scripts/cellprofiler.exe", "-c", "-r",
    "-p", "D:/CellPainting-Claw-main/CellPainting-Claw-main/demo/backend/profiling_backend/cellprofiler/CPJUMP1_analysis_smoketest.cppipe",
    "-o", str(OUT),
    "-i", "D:/CellPainting-Claw-main/CellPainting-Claw-main/demo/workspace/reference_data/BR00117035",
    "--data-file", "D:/CellPainting-Claw-main/CellPainting-Claw-main/demo/workspace/reference_data/BR00117035/load_data.csv",
    "-e", "imageio_reader_v3", "imageio_reader", "ngff_reader", "gcs_reader",
]

print(f"Start: {time.strftime('%H:%M:%S')}")
t0 = time.perf_counter()

modules = []
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    m = re.search(r"Image # (\d+), module (.+?) # (\d+): CPU_time = ([\d.]+) secs, Wall_time = ([\d.]+) secs", line)
    if m:
        img, name, mod_num, cpu, wall = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        modules.append((int(mod_num), name, float(cpu), float(wall)))
        print(f"  #{mod_num:2d} {name:<35s} CPU={cpu:>8s}s  Wall={wall:>8s}s")
    if "pipeline completed" in line:
        print("PIPELINE COMPLETED")

proc.wait()
elapsed = time.perf_counter() - t0

# Write report
report = OUT / "timing_report.md"
with open(report, "w") as f:
    f.write(f"# CellProfiler Optimized v4.2.8 — Module Timing Report\n\n")
    f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Data:** BR00117035 (1 site, 1080×1080×8ch)\n")
    f.write(f"**Pipeline:** CPJUMP1_analysis_smoketest.cppipe\n")
    f.write(f"**Total wall:** {elapsed:.1f}s  |  **Total CPU:** {sum(m[2] for m in modules):.1f}s  |  **Modules:** {len(modules)}\n\n")
    f.write("| # | Module | CPU (s) | Wall (s) | % CPU |\n")
    f.write("|---|--------|---------|----------|------|\n")
    total_cpu = sum(m[2] for m in modules)
    for num, name, cpu, wall in modules:
        pct = cpu / total_cpu * 100 if total_cpu > 0 else 0
        f.write(f"| {num} | {name} | {cpu:.2f} | {wall:.2f} | {pct:.1f}% |\n")
    f.write(f"\n## Summary\n\n")
    f.write(f"- **Total wall-clock:** {elapsed:.1f}s ({elapsed/60:.1f} min)\n")
    f.write(f"- **Total CPU:** {total_cpu:.1f}s\n")
    f.write(f"- **Exit code:** {proc.returncode}\n")
    # Top 5 slowest
    f.write(f"\n### Top 5 Slowest Modules\n\n")
    for num, name, cpu, wall in sorted(modules, key=lambda x: x[2], reverse=True)[:5]:
        f.write(f"- **{name}** (#{num}): {cpu:.1f}s CPU ({cpu/total_cpu*100:.0f}%)\n")

print(f"\nWall: {elapsed:.1f}s | Report: {report}")
print(f"Output CSVs: {list(OUT.glob('*.csv'))}")
