"""Automatic checks for each exercise. Every exercise calls its check_exN() when you run it.

Try not to read this file until you've finished an exercise: some checks build the
expected answer, so reading them gives the answer away.
"""
import math
import traceback

import numpy as np

import config

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"


class _Report:
    def __init__(self, title):
        self.title, self.passed, self.failed, self.todo = title, 0, 0, 0
        print(f"\n=== Checking {title} ===")

    def run(self, name, fn):
        try:
            fn()
        except NotImplementedError:
            self.todo += 1
            print(f"  {YELLOW}TODO{RESET}  {name}  (not implemented yet)")
        except AssertionError as e:
            self.failed += 1
            print(f"  {RED}FAIL{RESET}  {name}\n        {e}")
        except Exception:
            self.failed += 1
            print(f"  {RED}ERROR{RESET} {name}  (your code raised an exception)")
            print("        " + traceback.format_exc().strip().replace("\n", "\n        "))
        else:
            self.passed += 1
            print(f"  {GREEN}PASS{RESET}  {name}")

    def skip(self, name, why):
        print(f"  {YELLOW}SKIP{RESET}  {name}  ({why})")

    def done(self):
        total = self.passed + self.failed + self.todo
        ok = self.failed == 0 and self.todo == 0 and total > 0
        color = GREEN if ok else RED
        print(f"{color}{self.passed}/{total} checks passed{RESET}\n")
        return ok


def _close(a, b, tol=1e-3):
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=tol)


# ------------------------------------------------------------------ exercise 1
def check_ex1(describe_io, count_ops, find_nms_node):
    import onnx
    from collections import Counter

    r = _Report("exercise 1: inspect the ONNX model")
    model = onnx.load(str(config.ONNX_PATH))

    def shapes(values):
        return {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in values}

    want_in, want_out = shapes(model.graph.input), shapes(model.graph.output)

    def t_io():
        io = describe_io(model)
        assert isinstance(io, dict) and set(io) == {"inputs", "outputs"}, \
            'return a dict like {"inputs": {...}, "outputs": {...}}'
        got_in = {k: tuple(v) for k, v in io["inputs"].items()}
        got_out = {k: tuple(v) for k, v in io["outputs"].items()}
        assert got_in == want_in, f"inputs should be {want_in}, got {got_in}"
        assert set(got_out) == set(want_out), f"output names should be {sorted(want_out)}, got {sorted(got_out)}"
        assert got_out == want_out, "output names are right but some shapes are wrong"

    def t_ops():
        got = count_ops(model)
        want = Counter(n.op_type for n in model.graph.node)
        assert dict(got) == dict(want), \
            f"op counts don't match. Expected {len(want)} op types, {sum(want.values())} nodes total"

    def t_nms():
        op_type, attrs = find_nms_node(model)
        assert op_type == "EfficientNMS_TRT", f"expected the EfficientNMS_TRT node, got {op_type!r}"
        for key, val in [("box_coding", 0), ("max_output_boxes", 64)]:
            assert attrs.get(key) == val, f"attribute {key} should be {val}, got {attrs.get(key)!r}"
        assert math.isclose(attrs.get("score_threshold", -1), 0.7, abs_tol=1e-4), "score_threshold looks wrong"
        assert math.isclose(attrs.get("iou_threshold", -1), 0.3, abs_tol=1e-4), "iou_threshold looks wrong"

    r.run("describe_io() finds every input/output and its shape", t_io)
    r.run("count_ops() counts each op type", t_ops)
    r.run("find_nms_node() finds the NMS node and reads its attributes", t_nms)
    return r.done()


# ------------------------------------------------------------------ exercise 2
def check_ex2(TRTModel):
    r = _Report("exercise 2: load the TensorRT engine")
    try:
        import tensorrt  # noqa: F401
    except ImportError:
        r.skip("TRTModel", "TensorRT isn't installed here; run this on the Jetson")
        return True
    if not config.ENGINE_PATH.exists():
        r.skip("TRTModel", f"no engine at {config.ENGINE_PATH}; build it with trtexec first (see README)")
        return False

    state = {}

    def t_load():
        state["m"] = TRTModel(config.ENGINE_PATH)

    def t_io():
        info = state["m"].io_info()
        names = {row[0] for row in info}
        want = {"images", "output/sink_port_0", "num_detections", "detection_boxes",
                "detection_scores", "detection_classes"}
        assert names == want, f"expected tensors {sorted(want)}, got {sorted(names)}"
        shapes = {row[0]: tuple(row[2]) for row in info}
        assert shapes["images"] == (1, 3, 640, 640), f"input shape should be (1, 3, 640, 640), got {shapes['images']}"
        assert shapes["output/sink_port_0"] == (1, 25200, 22), "raw output shape looks wrong"

    def t_infer():
        out = state["m"](np.zeros((1, 3, 640, 640), np.float32))
        assert isinstance(out, dict), "__call__ should return a dict {name: numpy array}"
        assert "images" not in out, "only return outputs, not the input"
        raw = out["output/sink_port_0"]
        assert isinstance(raw, np.ndarray), "outputs should be numpy arrays (copy them back to the CPU)"
        assert raw.shape == (1, 25200, 22), f"raw output shape should be (1, 25200, 22), got {raw.shape}"
        assert np.isfinite(raw).all(), "raw output contains NaN/Inf"

    r.run("TRTModel loads the engine", t_load)
    if "m" in state:
        r.run("io_info() lists every tensor with its shape", t_io)
        r.run("calling the model returns all outputs as numpy arrays", t_infer)
    return r.done()


# ------------------------------------------------------------------ exercise 3
def check_ex3(letterbox, to_tensor):
    import cv2

    r = _Report("exercise 3: preprocessing")
    rng = np.random.default_rng(0)

    def t_wide():
        img = rng.integers(0, 256, (480, 1280, 3), dtype=np.uint8)
        out = letterbox(img)
        assert isinstance(out, tuple) and len(out) == 2, "letterbox() should return (canvas, scale)"
        canvas, scale = out
        assert canvas.shape == (640, 640, 3) and canvas.dtype == np.uint8, \
            f"canvas should be a 640x640x3 uint8 image, got {canvas.shape} {canvas.dtype}"
        assert math.isclose(scale, 0.5), f"for a 1280x480 image the scale should be 0.5, got {scale}"
        assert not canvas[240:].any(), \
            "everything below the resized image should be black padding. Is the image placed at the top-left?"
        ref = cv2.resize(img, (640, 240))
        diff = np.abs(canvas[:240].astype(int) - ref.astype(int)).mean()
        assert diff < 2, "the top 240 rows don't look like the resized image (or it isn't at the top-left)"

    def t_tall():
        img = rng.integers(0, 256, (1000, 500, 3), dtype=np.uint8)
        canvas, scale = letterbox(img)
        assert math.isclose(scale, 0.64), f"for a 500x1000 image the scale should be 0.64, got {scale}"
        assert not canvas[:, 320:].any(), "for a tall image the padding should be on the right"

    def t_tensor():
        canvas = np.zeros((640, 640, 3), np.uint8)
        canvas[..., 0] = 255  # pure blue in OpenCV's BGR order
        x = to_tensor(canvas)
        assert isinstance(x, np.ndarray) and x.dtype == np.float32, "return a float32 numpy array"
        assert x.shape == (1, 3, 640, 640), f"shape should be (1, 3, 640, 640), got {x.shape}"
        assert x.max() <= 1.0 + 1e-6, "values should be scaled to 0..1"
        assert math.isclose(float(x[0, 2].mean()), 1.0) and float(x[0, 0].mean()) == 0.0, \
            "a pure blue image should end up in channel 2 (the model wants RGB, OpenCV gives BGR)"

    r.run("letterbox() on a wide image", t_wide)
    r.run("letterbox() on a tall image", t_tall)
    r.run("to_tensor() layout, dtype, range and color order", t_tensor)
    return r.done()


# ------------------------------------------------------------------ exercise 4
def _fake_row(corners_raw_order, score, color_id, number_id):
    row = np.full(22, -6.0, np.float32)
    row[0:8] = np.asarray(corners_raw_order, np.float32).reshape(-1)
    row[8] = math.log(score / (1 - score))
    row[9 + color_id] = 5.0
    row[13 + number_id] = 5.0
    return row


def check_ex4(sigmoid, decode_row):
    r = _Report("exercise 4: decode one row of the raw output")
    # corners in the model's raw order (TL, BL, BR, TR), in 640x640 input pixels
    raw = [(100, 50), (100, 80), (200, 82), (200, 48)]
    row = _fake_row(raw, 0.9, color_id=1, number_id=3)
    scale = 0.5
    want = np.array([(200, 100), (400, 96), (400, 164), (200, 160)], np.float32)  # TL, TR, BR, BL

    def t_sigmoid():
        assert math.isclose(float(sigmoid(0.0)), 0.5), "sigmoid(0) should be 0.5"
        with np.errstate(over="raise"):
            v = sigmoid(np.array([-50.0, 0.0, 50.0]))
        assert _close(v, [0, 0.5, 1]), "sigmoid should work on arrays"

    def t_score():
        d = decode_row(row, scale)
        assert isinstance(d, dict), "decode_row() should return a dict"
        s = d.get("score")
        assert s is not None, 'missing "score"'
        assert not math.isclose(s, row[8], abs_tol=1e-3), \
            "score is the raw value from column 8. It's a logit; what turns a logit into a probability?"
        assert math.isclose(s, 0.9, abs_tol=1e-3), f"score should be 0.9 for this row, got {s}"

    def t_labels():
        d = decode_row(row, scale)
        assert d.get("color") == "red" and d.get("color_id") == 1, \
            f'expected color "red" (id 1), got {d.get("color")!r} (id {d.get("color_id")!r})'
        assert d.get("number") == "3" and d.get("number_id") == 3, \
            f'expected number "3" (id 3), got {d.get("number")!r} (id {d.get("number_id")!r})'

    def t_corners():
        d = decode_row(row, scale)
        c = np.asarray(d.get("corners"), np.float32)
        assert c.shape == (4, 2), f'"corners" should be a 4x2 array, got shape {c.shape}'
        if _close(c, np.asarray(raw) / scale) or _close(c, raw):
            raise AssertionError("corners are still in the model's raw order. The PnP step wants "
                                 "top-left, top-right, bottom-right, bottom-left")
        if _close(c, want * scale):
            raise AssertionError("order is right but the points are still in 640x640 input pixels; "
                                 "map them back to the original image with the letterbox scale")
        assert _close(c, want), f"expected corners {want.tolist()}, got {c.tolist()}"

    r.run("sigmoid()", t_sigmoid)
    r.run("decode_row() score", t_score)
    r.run("decode_row() color and number", t_labels)
    r.run("decode_row() corners: order and scale", t_corners)
    return r.done()


# ------------------------------------------------------------------ exercise 5
def check_ex5(decode_all):
    r = _Report("exercise 5: decode every detection + NMS")
    out = np.full((25200, 22), -6.0, np.float32)
    plate_a = [(100, 100), (100, 140), (220, 142), (220, 98)]
    plate_a2 = [(103, 101), (103, 141), (223, 143), (223, 99)]  # near-duplicate of A
    plate_b = [(400, 300), (400, 330), (480, 331), (480, 299)]
    plate_c = [(50, 500), (50, 530), (120, 531), (120, 499)]  # below threshold
    out[100] = _fake_row(plate_a, 0.95, 0, 1)
    out[101] = _fake_row(plate_a2, 0.90, 0, 1)
    out[5000] = _fake_row(plate_b, 0.80, 1, 6)
    out[7000] = _fake_row(plate_c, 0.50, 1, 2)

    def run():
        dets = decode_all(out, 1.0)
        assert isinstance(dets, list), "decode_all() should return a list of dicts"
        return dets

    def t_count():
        dets = run()
        assert len(dets) != 4, "nothing was filtered. Drop rows below the score threshold and run NMS"
        assert len(dets) != 3, "got 3 detections: either the low-score row or the duplicate survived"
        assert len(dets) == 2, f"expected 2 detections, got {len(dets)}"

    def t_content():
        dets = sorted(run(), key=lambda d: -d["score"])
        a, b = dets
        assert math.isclose(a["score"], 0.95, abs_tol=1e-3), "the highest-scoring duplicate should survive NMS"
        assert a["color"] == "blue" and a["number"] == "1", "first plate should be blue 1"
        assert b["color"] == "red" and b["number"] == "outpost", "second plate should be red outpost"
        want_a = np.array([plate_a[0], plate_a[3], plate_a[2], plate_a[1]], np.float32)
        assert _close(a["corners"], want_a), "corners of the surviving plate are wrong"

    r.run("filters low scores and suppresses duplicates", t_count)
    r.run("returns the right plates with the right labels and corners", t_content)
    return r.done()


# ------------------------------------------------------------------ exercise 7
def check_ex7(armor_object_points, solve_pnp):
    import cv2

    r = _Report("exercise 7: distance with solvePnP")
    K = np.array(config.CAMERA_MATRIX, np.float64)
    dist = np.array(config.DIST_COEFFS, np.float64)

    def truth_points(big):
        w = config.BIG_ARMOR_WIDTH if big else config.SMALL_ARMOR_WIDTH
        h = config.LIGHTBAR_LENGTH
        return np.array([[0, w / 2, h / 2], [0, -w / 2, h / 2], [0, -w / 2, -h / 2], [0, w / 2, -h / 2]], np.float64)

    def t_points():
        for big in (False, True):
            p = np.asarray(armor_object_points(big), np.float64)
            assert p.shape == (4, 3), f"object points should be 4x3, got {p.shape}"
            w = config.BIG_ARMOR_WIDTH if big else config.SMALL_ARMOR_WIDTH
            top, side = np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[3])
            assert math.isclose(top, w, abs_tol=1e-6), \
                f"TL->TR should be the plate width ({w} m) for {'big' if big else 'small'} plates, got {top:.4f}"
            assert math.isclose(side, config.LIGHTBAR_LENGTH, abs_tol=1e-6), \
                f"TL->BL should be the light bar length ({config.LIGHTBAR_LENGTH} m), got {side:.4f}"

    def t_distance():
        # plate facing the camera, a little off-center and turned 20 degrees
        R_face = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], np.float64)
        yaw = math.radians(20)
        R_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
        rvec, _ = cv2.Rodrigues(R_yaw @ R_face)
        for big, tvec in [(False, [0.3, -0.1, 3.0]), (True, [-0.5, 0.2, 5.0])]:
            img_pts, _ = cv2.projectPoints(truth_points(big), rvec, np.array(tvec, np.float64), K, dist)
            got = solve_pnp(img_pts.reshape(4, 2).astype(np.float32), big, K, dist)
            assert isinstance(got, tuple) and len(got) == 2, "solve_pnp() should return (rvec, tvec)"
            t = np.asarray(got[1], np.float64).reshape(3)
            want = np.linalg.norm(tvec)
            assert math.isclose(np.linalg.norm(t), want, rel_tol=0.02), \
                f"plate is {want:.2f} m away but you got {np.linalg.norm(t):.2f} m " \
                "(check the point order and the plate size)"

    r.run("armor_object_points() has the right size and order", t_points)
    r.run("solve_pnp() recovers the true distance", t_distance)
    return r.done()
