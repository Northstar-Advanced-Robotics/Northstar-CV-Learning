"""Exercise 1: look inside the model file before you ever run it.

Goal: find out what goes into the model, what comes out, and what the extra "NMS" step that was
added to it does (NMS removes duplicate guesses: BASICS.md, section 7).

How an ONNX file is organized (onnx.load() gives you this as Python objects):
    model.graph.input    the inputs:  each has a name and a shape, e.g. "images", [1, 3, 640, 640]
    model.graph.output   the outputs: same idea
    model.graph.node     every operation ("node") in order. node.op_type is what it does
                         ("Conv", "Sigmoid", ...), node.attribute holds its settings

Fill in the three functions below, then run:  python3 ex1_inspect_onnx.py
Useful docs: https://onnx.ai/onnx/intro/python.html  (and try dragging the .onnx file onto https://netron.app)
"""
from collections import Counter  # noqa: F401  (you'll probably want this)

import onnx

import checks
import config


def describe_io(model):
    """Return {"inputs": {name: shape}, "outputs": {name: shape}} for the model's graph.

    Example of the format: {"inputs": {"images": [1, 3, 640, 640]}, "outputs": {...}}
    Start by exploring in a Python shell:
        import onnx, config
        model = onnx.load(str(config.ONNX_PATH))
        print(model.graph.input[0])      # what's inside it?
    """
    # TODO
    raise NotImplementedError


def count_ops(model):
    """Return a dict {op_type: how many nodes of that type}, e.g. {"Conv": 61, ...}.

    The operations are in model.graph.node; each one's kind is node.op_type.
    collections.Counter counts things for you: Counter(["a", "b", "a"]) == {"a": 2, "b": 1}.
    """
    # TODO
    raise NotImplementedError


def find_nms_node(model):
    """Find the NMS (non-maximum suppression) node: the one whose op_type contains "NMS".
    Return (op_type, {setting_name: value}), e.g. ("EfficientNMS_TRT", {"iou_threshold": 0.3, ...}).

    A node's settings are in node.attribute: each one has a .name, and
    onnx.helper.get_attribute_value(attr) turns it into a normal Python value.
    """
    # TODO
    raise NotImplementedError


def producers(model):
    """Map each tensor name to the node that produces it, so you can walk the graph backwards."""
    return {out: node for node in model.graph.node for out in node.output}


if __name__ == "__main__":
    model = onnx.load(str(config.ONNX_PATH))
    if not checks.check_ex1(describe_io, count_ops, find_nms_node):
        raise SystemExit

    io = describe_io(model)
    print("Inputs: ", io["inputs"])
    print("Outputs:")
    for name, shape in io["outputs"].items():
        print(f"   {name:22s} {shape}")

    print("\nMost common ops:", Counter(count_ops(model)).most_common(8))

    op_type, attrs = find_nms_node(model)
    print(f"\nNMS node: {op_type}")
    for k, v in attrs.items():
        print(f"   {k} = {v}")

    # Walk backwards from the NMS node's box input to see how its boxes were built
    nms = next(n for n in model.graph.node if n.op_type == op_type)
    prod = producers(model)
    boxes_node = prod[nms.input[0]]
    print(f"\nNMS box input '{nms.input[0]}' is made by a {boxes_node.op_type} of: {list(boxes_node.input)}")
    print(f"NMS score input '{nms.input[1]}' is made by a {prod[nms.input[1]].op_type}")
