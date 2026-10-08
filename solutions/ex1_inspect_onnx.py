"""Exercise 1: look inside the ONNX model before you ever run it.

Run:  python3 ex1_inspect_onnx.py
"""
from collections import Counter

import onnx

import checks
import config


def describe_io(model):
    """Return {"inputs": {name: shape}, "outputs": {name: shape}} for the model's graph."""
    def shapes(values):
        return {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in values}

    return {"inputs": shapes(model.graph.input), "outputs": shapes(model.graph.output)}


def count_ops(model):
    """Return a dict {op_type: how many nodes of that type} (e.g. {"Conv": 61, ...})."""
    return Counter(node.op_type for node in model.graph.node)


def find_nms_node(model):
    """Find the non-maximum-suppression node. Return (op_type, {attribute_name: value})."""
    for node in model.graph.node:
        if "NMS" in node.op_type:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            return node.op_type, attrs
    return None, {}


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

    print("\nMost common ops:", count_ops(model).most_common(8))

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
