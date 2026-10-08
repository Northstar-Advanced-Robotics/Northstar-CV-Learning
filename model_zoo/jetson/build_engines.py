"""Build a TensorRT engine for every ONNX model in the zoo, on THIS device. Run it on the Jetson.

    python3 jetson/build_engines.py                        # every ONNX model, FP16
    python3 jetson/build_engines.py --models rp_0526,tongji_yolo11
    python3 jetson/build_engines.py --fp32                 # without --fp16, to compare speed
    python3 jetson/build_engines.py --force                # rebuild engines that already exist

Engines land in model_zoo/engines/<model>.engine (or <model>_fp32.engine) and appear in zoo.py as
<model>_trt (or <model>_trt_fp32), so m2-m8 run them like any other file. One build takes a few
minutes on an Orin Nano: close other GPU programs first. trtexec's full output goes to
engines/<model>.log; the GPU time it measured is printed at the end.

Why build here and not copy one? An engine is compiled for one GPU and one TensorRT version
(look at the engines in models/: they say TensorRT 10.6, JetPack 6 has 10.3). The ONNX is the
portable file; the engine is not.
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zoo  # noqa: E402

GPU_TIME = re.compile(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms")


def find_trtexec():
    for candidate in (shutil.which("trtexec"), "/usr/src/tensorrt/bin/trtexec"):
        if candidate and Path(candidate).exists():
            return candidate
    raise SystemExit("trtexec not found. On a Jetson it's in /usr/src/tensorrt/bin/ (part of JetPack).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated ONNX model names from zoo.py (default: all of them)")
    ap.add_argument("--fp32", action="store_true", help="build without --fp16")
    ap.add_argument("--force", action="store_true", help="rebuild even if the engine exists")
    args = ap.parse_args()

    trtexec = find_trtexec()
    zoo.ENGINES_DIR.mkdir(exist_ok=True)
    models = zoo.find(args.models, formats=("onnx",))
    if not models:
        raise SystemExit("No ONNX models found.")
    suffix = "_fp32" if args.fp32 else ""
    summary = []
    for m in models:
        engine = zoo.ENGINES_DIR / f"{m.name}{suffix}.engine"
        log = zoo.ENGINES_DIR / f"{m.name}{suffix}.log"
        if engine.exists() and not args.force:
            print(f"{m.name}: {engine.name} already exists (--force to rebuild)")
            summary.append((m.name, "exists", log))
            continue
        cmd = [trtexec, f"--onnx={m.file}", f"--saveEngine={engine}"] + ([] if args.fp32 else ["--fp16"])
        print(f"\n{m.name}: {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, capture_output=True, text=True)
        log.write_text(result.stdout + result.stderr)
        if result.returncode != 0 or not engine.exists():
            tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-8:])
            print(f"  FAILED (full log: {log})\n{tail}")
            summary.append((m.name, "FAILED", log))
            continue
        summary.append((m.name, "built", log))
        print(f"  built {engine.name}")

    print(f"\n{'model':26s} {'status':8s} {'GPU median ms':>14s}")
    for name, status, log in summary:
        found = GPU_TIME.search(log.read_text()) if log.exists() else None
        print(f"{name:26s} {status:8s} {found.group(4) if found else '-':>14s}")
    print("\n(GPU time is the network alone. m8_benchmark.py measures the whole frame: copy, preprocess, decode.)")


if __name__ == "__main__":
    main()
