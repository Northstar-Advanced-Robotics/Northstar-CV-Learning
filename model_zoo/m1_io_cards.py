"""m1: what does each model file say about itself? Nothing runs here: we only READ the files.

Before you can feed a model anything you need to know its inputs and outputs ("IO"): each one's
name, shape and number type ("dtype": float32, float16, ...). Some of that is written in the file. Some of it never is (RGB or
BGR? 0..1 or 0..255? where does the padding go?), and some files carry extra notes: class labels,
the image size they were trained at, which file they were converted from, whether they're INT8.

Run:  python3 m1_io_cards.py                          # a card for every model file in zoo.py
      python3 m1_io_cards.py --models rp_0708,rp_0708_openvino
      python3 m1_io_cards.py --markdown io_cards.md     # also save the summary table
"""
import argparse
from collections import Counter

import checks
import runtimes
import zoo


def onnx_card(path):
    """Read an .onnx file with the `onnx` package (NOT onnxruntime: we don't run it). Return
        {"inputs":   [(name, shape, dtype), ...]   e.g. ("images", [1, 3, 640, 640], "float32")
         "outputs":  [(name, shape, dtype), ...]
         "producer": "pytorch 2.0.1"               who wrote the file
         "ops":      Counter({"Conv": 61, ...})     what the graph is made of
         "metadata": {key: value}}                 extra notes stored in the file
    """
    # onnx.load(path) gives a ModelProto. Everything is in there:
    #   model.graph.input / model.graph.output: each has .name and .type.tensor_type, which has
    #     .elem_type (a number: onnx.helper.tensor_dtype_to_np_dtype(n).name turns it into "float32")
    #     .shape.dim (each dim has .dim_value, or .dim_param if it's a named, variable size)
    #   model.graph.node: every operation, .op_type is its kind ("Conv", "Sigmoid", ...)
    #   model.producer_name / .producer_version, model.metadata_props (each has .key and .value)
    # Old exporters also list the weights in graph.input: skip names that are in graph.initializer.
    import onnx  # imported here so the other scripts work without the onnx package

    # TODO
    raise NotImplementedError


def openvino_card(path):
    """Read an OpenVINO .xml (+ .bin) without compiling it. Same dict as onnx_card()."""
    # openvino.Core().read_model(path) gives a Model:
    #   model.inputs / model.outputs: each port has .get_names() (a set: it can be EMPTY),
    #     .get_partial_shape() (iterate it; each dim has .is_static and .get_length()),
    #     .get_element_type().to_dtype().name -> "float32"
    #   model.get_ordered_ops(): every op, .get_type_name() is its kind ("Convolution", ...)
    #   rt_info is OpenVINO's metadata. Read one entry with model.has_rt_info(["model_info", "labels"])
    #     and model.get_rt_info(["model_info", "labels"]).astype(str). runtimes.OpenVinoRunner.RT_INFO
    #     lists the entries worth reading; "Runtime_version" makes a good "producer".
    # Use None as the name of a port that has no names.
    import openvino as ov

    # TODO
    raise NotImplementedError


def ncnn_card(path, input_hw=None):
    """NCNN .param files are plain text: one layer per line. Open one in an editor and look."""
    lines = path.read_text().splitlines()
    ins, outs = runtimes._ncnn_blobs(path)
    hw = list(input_hw) if input_hw else ["?", "?"]  # from zoo.py: the file doesn't say
    return {"inputs": [(n, [1, 3] + hw, "float32") for n in ins],
            "outputs": [(n, ["?"], "float32") for n in outs],
            "producer": "NCNN " + ("(pnnx)" if path.name.endswith(".ncnn.param") else "(onnx2ncnn)"),
            "ops": Counter(line.split()[0] for line in lines[2:] if line.strip()),
            "metadata": {"note": "a .param file stores no input size and no dtype: you have to know them"}}


def engine_card(path):
    """A TensorRT engine is a compiled program for ONE GPU + TensorRT version. Reading its IO means
    loading it, which only works where it was built."""
    built = runtimes.engine_trt_version(path)
    card = {"inputs": [], "outputs": [], "producer": f"TensorRT {built or '?'}", "ops": Counter(), "metadata": {}}
    try:
        r = runtimes.TensorRTRunner(path)
        card["inputs"] = [(t.name, t.shape, t.dtype.name) for t in r.inputs]
        card["outputs"] = [(t.name, t.shape, t.dtype.name) for t in r.outputs]
    except ImportError:
        card["metadata"]["note"] = "TensorRT isn't installed here, so the IO can't be read (try on the Jetson)"
    except RuntimeError as e:
        card["metadata"]["note"] = str(e)
    return card


def card_for(m):
    if m.format == "ncnn":
        return ncnn_card(m.file, m.input_hw)
    return {"onnx": onnx_card, "openvino": openvino_card, "tensorrt": engine_card}[m.format](m.file)


def flags(m, card):
    """Things worth noticing, spotted from the card."""
    out = []
    if card["ops"].get("EfficientNMS_TRT"):
        out.append("contains EfficientNMS_TRT: only TensorRT can run it")
    if any(dtype == "float16" for _, _, dtype in card["inputs"]):
        out.append("FLOAT16 input")
    if card["ops"].get("FakeQuantize") or "nncf" in str(card["metadata"]):
        out.append("INT8 (quantized)")
    if any(name is None for name, _, _ in card["outputs"]):
        out.append("an output has NO name: use it by position")
    if m.format == "ncnn" and m.input_hw:
        out.append(f"input size isn't in the file; zoo.py says {m.input_hw[0]}x{m.input_hw[1]}")
    if m.format == "tensorrt":
        out.append(f"{card['producer']} built it: it only loads with that version")
    return out


def print_card(m, card):
    print(f"\n=== {m.name}  ({m.file.relative_to(zoo.MODELS_DIR) if m.file.is_relative_to(zoo.MODELS_DIR) else m.file})")
    print(f"    format {m.format}, family {m.family}, network {m.network}, from {m.team}")
    print(f"    producer: {card['producer']}")
    for kind in ("inputs", "outputs"):
        for name, shape, dtype in card[kind]:
            print(f"    {kind[:-1]:6s} {name if name is not None else '(no name)':22s} {str(shape):18s} {dtype}")
    if card["ops"]:
        print(f"    {sum(card['ops'].values())} ops, most common: "
              + ", ".join(f"{k} {v}" for k, v in card["ops"].most_common(5)))
    for k, v in card["metadata"].items():
        v = " ".join(str(v).split())
        print(f"    meta {k}: {v[:110] + '...' if len(v) > 110 else v}")
    for f in flags(m, card):
        print(f"    !! {f}")
    print(f"    notes: {m.notes}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", help="comma-separated names from zoo.py (default: all)")
    ap.add_argument("--markdown", help="also write the summary table to this file")
    args = ap.parse_args()

    if not checks.check_m1(onnx_card, openvino_card):
        raise SystemExit
    rows = []
    for m in zoo.find(args.models):
        try:
            card = card_for(m)
        except Exception as e:
            print(f"\n=== {m.name}: couldn't read it ({e})")
            continue
        print_card(m, card)
        ins = "; ".join(f"{n} {s} {d}" for n, s, d in card["inputs"]) or "?"
        outs = "; ".join(f"{n or '(no name)'} {s}" for n, s, _ in card["outputs"]) or "?"
        rows.append((m.name, m.format, m.family, ins, outs, ", ".join(flags(m, card))))

    header = ("model", "format", "family", "input", "output(s)", "watch out")
    table = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    table += ["| " + " | ".join(r) + " |" for r in rows]
    print("\n" + "\n".join(table))
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(table) + "\n")
        print(f"\nSaved {args.markdown}")


if __name__ == "__main__":
    main()
