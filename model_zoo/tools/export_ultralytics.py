"""Turn an Ultralytics checkpoint (.pt) into an ONNX file the zoo can run and the Jetson can build an engine from.

    python3 tools/export_ultralytics.py ../models/northstar/ns_yolo26n_v2.pt
    python3 tools/export_ultralytics.py my_model.pt --imgsz 640 --out ../models/northstar/my_model.onnx

Needs the official `ultralytics` package, ideally the version the checkpoint was saved with (this script
prints it): pip install ultralytics==8.4.187 onnx. Do it on a laptop, in its own virtual environment.

A .pt file is a Python pickle: loading one runs code. Only export checkpoints you trust: check first with
tools/inspect_pt.py, which reads a .pt without loading it.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")  # don't let ultralytics pip-install things on its own


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt", help="the .pt checkpoint")
    ap.add_argument("--imgsz", type=int, help="input size (default: the size it was trained at)")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset (17 works with TensorRT 10.3 on JetPack 6)")
    ap.add_argument("--out", help="where to put the .onnx (default: next to the .pt, same name)")
    args = ap.parse_args()

    from ultralytics import YOLO, __version__

    pt = Path(args.pt).resolve()
    model = YOLO(str(pt))
    saved_with = (model.ckpt or {}).get("version", "?")
    imgsz = args.imgsz or (model.ckpt.get("train_args") or {}).get("imgsz", 640)
    print(f"{pt.name}: task {model.task}, saved with ultralytics {saved_with} (you have {__version__}), "
          f"input {imgsz}, classes {model.names}")
    # nms=False asks for the NMS-free ("end-to-end") head when the model has one (YOLO26 does): the model
    # then removes its own duplicates. Without it, ultralytics 8.4 exports the raw head instead, a
    # completely different output: for our YOLO26 model (1, 24, 8400) instead of (1, 300, 18).
    onnx_file = Path(model.export(format="onnx", imgsz=imgsz, opset=args.opset, simplify=False, dynamic=False,
                                  nms=False))
    out = Path(args.out) if args.out else pt.with_suffix(".onnx")
    if onnx_file.resolve() != out.resolve():
        shutil.copy(onnx_file, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    sys.exit(main())
