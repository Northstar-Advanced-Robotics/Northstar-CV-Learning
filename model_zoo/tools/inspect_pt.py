"""Look inside a PyTorch checkpoint (.pt) WITHOUT loading it, so none of its code can run.

    python3 tools/inspect_pt.py ../models/northstar/ns_yolo26n_v2.pt

A .pt file is a zip archive holding a Python "pickle", and loading a pickle can run any code its author put
in it. This script never loads it for real. It prints:
  1. every Python module the pickle would import. Expect only torch, ultralytics, collections, numpy and
     Python's own types. Anything like os, subprocess or socket means: don't load it.
  2. the training settings it remembers (Ultralytics checkpoints): task, classes, corner format, picture size,
     dataset, dates. To read them, every class in the pickle is replaced with a harmless placeholder.
Only needs Python itself: no torch, no ultralytics.
"""
import pickle
import pickletools
import sys
import zipfile
from collections import Counter

SAFE_MODULES = {"torch", "ultralytics", "collections", "numpy", "_codecs", "builtins", "__builtin__", "dill", "pathlib",
                "datetime"}
DANGEROUS = {"os", "subprocess", "sys", "shutil", "socket", "requests", "urllib", "webbrowser", "nt", "posix"}


def imports(data):
    """module.name of everything the pickle would import, read straight from its instructions."""
    refs, last = Counter(), []
    for op, arg, _ in pickletools.genops(data):
        if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
            last = (last + [arg])[-2:]
        elif op.name == "STACK_GLOBAL" and len(last) == 2:
            refs[f"{last[0]}.{last[1]}"] += 1
        elif op.name in ("GLOBAL", "INST"):
            refs[str(arg).replace(" ", ".")] += 1
    return refs


class Placeholder:
    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


class PlaceholderUnpickler(pickle.Unpickler):
    """Builds plain Python containers for real, and a Placeholder for every other class: nothing executes."""
    BUILTINS = {"set": set, "frozenset": frozenset, "dict": dict, "list": list, "tuple": tuple}

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            from collections import OrderedDict
            return OrderedDict
        if module in ("builtins", "__builtin__") and name in self.BUILTINS:
            return self.BUILTINS[name]
        return Placeholder

    def persistent_load(self, pid):
        return "<weights>"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    z = zipfile.ZipFile(sys.argv[1])
    name = next(n for n in z.namelist() if n.endswith("data.pkl"))
    data = z.read(name)

    refs = imports(data)
    modules = Counter(r.split(".")[0] for r in refs)
    print(f"{sys.argv[1]} ({name})")
    print("modules it would import:", dict(modules))
    bad = sorted(r for r in refs if r.split(".")[0] in DANGEROUS or r.split(".")[-1] in ("system", "exec", "eval", "popen"))
    unusual = sorted(r for r in refs if r.split(".")[0] not in SAFE_MODULES)
    print("DANGEROUS, don't load it:" if bad else "nothing dangerous found", bad or "")
    if unusual:
        print("unusual references (find out what these are before loading):", unusual)
    model_classes = sorted(r for r in refs if r.startswith("ultralytics.nn.tasks"))
    if model_classes:
        print("model class:", ", ".join(model_classes))

    ckpt = PlaceholderUnpickler(z.open(name)).load()
    if not isinstance(ckpt, dict):
        return
    print("\nsaved:", ckpt.get("date"), "with ultralytics", ckpt.get("version"))
    args = ckpt.get("train_args") or {}
    for k in ("task", "model", "data", "imgsz", "epochs", "batch"):
        if k in args:
            print(f"  {k}: {args[k]}")
    model = ckpt.get("ema") or ckpt.get("model")
    state = getattr(model, "state", None) or {}
    yaml = state.get("yaml") if isinstance(state, dict) else None
    if isinstance(yaml, dict):
        print("  classes:", state.get("names"))
        print("  corners (kpt_shape):", yaml.get("kpt_shape"), " end-to-end head:", yaml.get("end2end", False))


if __name__ == "__main__":
    main()
