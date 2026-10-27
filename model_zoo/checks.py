"""Automatic checks for the model zoo. Every script calls its checks when you run it.

Like the main track: try not to read this until you're done, the fake outputs below give the
layouts away.
"""
import math
import traceback

import numpy as np

import zoo
from common import Transform
from runtimes import Tensor

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
        ok = self.failed == 0 and self.todo == 0
        print(f"{GREEN if ok else RED}{self.passed}/{total} checks passed{RESET}\n")
        return ok


def _close(a, b, tol=1e-3):
    return np.allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), atol=tol)


def _frame(h, w, bgr=(255, 0, 0)):
    """A solid-color frame (default pure blue in OpenCV's BGR order) with a white marker top-left."""
    f = np.zeros((h, w, 3), np.uint8)
    f[:] = bgr
    f[:4, :4] = 255
    return f


def _check_tensor(x, tf, shape, dtype):
    assert isinstance(x, np.ndarray), "preprocess() should return (numpy array, Transform)"
    assert isinstance(tf, Transform), "the second thing preprocess() returns should be letterbox()'s Transform"
    assert x.shape == shape, f"tensor shape should be {shape}, got {x.shape}"
    assert x.dtype == dtype, f"tensor dtype should be {np.dtype(dtype)}, got {x.dtype}"


def _corners(dets, k=0):
    return np.asarray(dets[k]["corners"], np.float32)


# ============================================================================ Part A (before decoding)
def check_part_a(family, module):
    r = _Report(f"{family}: preprocess and the helpers Part A needs")
    {"yolov5": _a_yolov5, "yolox": _a_yolox, "ultralytics": _a_ultralytics, "szurp_v8": _a_szurp_v8,
     "end2end": _a_end2end, "yolo26_e2e": _a_yolo26}[family](r, module)
    return r.done()


def _a_yolov5(r, m):
    def t_layout():
        x, tf = m.preprocess(_frame(480, 1280), Tensor("images", [1, 3, 640, 640], np.dtype(np.float32)))
        _check_tensor(x, tf, (1, 3, 640, 640), np.float32)
        assert math.isclose(tf.scale, 0.5) and tf.pad_x == 0 and tf.pad_y == 0, \
            "the image should sit in the TOP-LEFT corner (pad 0, 0), scaled by 0.5"
        assert x.max() <= 1.0 + 1e-6, "values should be 0..1"
        assert math.isclose(float(x[0, 2, 100:200].mean()), 1.0) and float(x[0, 0, 100:200].mean()) == 0.0, \
            "pure blue should end up in channel 2: the model wants RGB"
        assert not x[0, :, 240:].any(), "below the image should be black padding"

    def t_fp16():
        x, _ = m.preprocess(_frame(480, 1280), Tensor("images", [1, 3, 640, 640], np.dtype(np.float16)))
        assert x.dtype == np.float16, f"rp_0526 wants float16: use the dtype in inp, got {x.dtype}"

    def t_raw():
        raw = np.arange(25200 * 22, dtype=np.float32).reshape(1, 25200, 22)
        extra = [np.array([[3]], np.int32), raw, np.zeros((1, 64, 4), np.float32), np.zeros((1, 64), np.float32)]
        got = m.raw_output(extra)
        assert isinstance(got, np.ndarray) and got.shape == (25200, 22), \
            f"raw_output() should return the (25200, 22) array, got {getattr(got, 'shape', got)}"
        assert _close(got[5], raw[0, 5]), "that's the right shape but not the right array"

    r.run("preprocess(): letterbox, color order, range", t_layout)
    r.run("preprocess(): uses the model's dtype (float16 for rp_0526)", t_fp16)
    r.run("raw_output(): finds the raw output by shape, among NMS outputs too", t_raw)


def _a_yolox(r, m):
    def t_layout():
        f = _frame(540, 720)
        x, tf = m.preprocess(f, Tensor("images", [1, 3, 416, 416], np.dtype(np.float32)))
        _check_tensor(x, tf, (1, 3, 416, 416), np.float32)
        assert tf.pad_x == 0 and tf.pad_y == 52, \
            f"a 720x540 frame should be CENTERED: 52 rows of padding on top, got pad ({tf.pad_x}, {tf.pad_y})"
        assert x.max() > 200, "TUP feeds raw 0..255 values: don't divide by 255"
        assert float(x[0, 0, 100:300].mean()) == 255.0 and float(x[0, 2, 100:300].mean()) == 0.0, \
            "TUP keeps OpenCV's BGR order: pure blue should stay in channel 0"
        assert not x[0, :, :52].any() and not x[0, :, 364:].any(), "the bars above and below should be black (0)"

    def t_rows():
        a = np.random.default_rng(0).random((1, 3549, 21)).astype(np.float32)
        assert m.as_rows(a).shape == (3549, 21), "as_rows() should return (3549, 21)"
        assert _close(m.as_rows(a.transpose(0, 2, 1)), a[0]), \
            "given the pnnx layout (1, 21, 3549), as_rows() should transpose it back to (3549, 21)"

    def t_grid():
        g = np.asarray(m.grid_strides(416, 416))
        assert g.shape == (3549, 3), f"grid_strides(416, 416) should be (3549, 3), got {g.shape}"
        for row, want in [(0, (0, 0, 8)), (1, (1, 0, 8)), (52, (0, 1, 8)), (2704, (0, 0, 16)),
                          (2704 + 27, (1, 1, 16)), (3548, (12, 12, 32))]:
            assert _close(g[row], want), f"row {row} should be (x, y, stride) = {want}, got {g[row].tolist()}"

    def t_corners():
        rows = np.zeros((2, 21), np.float32)
        rows[0, :8] = [0.5, 0.5, 0, 1, 1, 1, 1, 0]
        rows[1, :8] = [-1, -1, 2, 3, 0, 0, 0, 0]
        grids = np.array([[2, 3, 8], [1, 1, 32]], np.float32)
        got = np.asarray(m.corners_in_input_pixels(rows, grids))
        want = [[[20, 28], [16, 32], [24, 32], [24, 24]], [[0, 0], [96, 128], [32, 32], [32, 32]]]
        assert got.shape == (2, 4, 2), f"should return (N, 4, 2), got {got.shape}"
        assert _close(got, want), "pixel = (value + cell) * stride, for x with the cell's x and y with its y"

    r.run("preprocess(): centered letterbox, BGR, 0..255", t_layout)
    r.run("as_rows(): both output orientations", t_rows)
    r.run("grid_strides(): one (x, y, stride) per row, in model order", t_grid)
    r.run("corners_in_input_pixels(): cell offsets -> pixels", t_corners)


def _a_ultralytics(r, m):
    def t_layout():
        x, tf = m.preprocess(_frame(540, 720), Tensor("images", [1, 3, 640, 640], np.dtype(np.float32)))
        _check_tensor(x, tf, (1, 3, 640, 640), np.float32)
        assert tf.pad_x == 0 and tf.pad_y == 80, \
            f"a 720x540 frame should be CENTERED in 640x640 (80 rows of bars on top), got pad ({tf.pad_x}, {tf.pad_y})"
        assert x.max() <= 1.0 + 1e-6, "values should be 0..1 (scale_values 255 means: divide by 255)"
        assert math.isclose(float(x[0, 2, 200:400].mean()), 1.0), \
            "pure blue should end up in channel 2 (reverse_input_channels: the network wants RGB)"
        assert _close(x[0, :, :80].mean(), 114 / 255), "the bars should be gray 114 (pad_value in the metadata)"

    def t_rows():
        a = np.random.default_rng(0).random((1, 50, 8400)).astype(np.float32)
        got = m.as_rows(a)
        assert got.shape == (8400, 50) and _close(got[7], a[0, :, 7]), "as_rows() should give (8400, 50): one row per candidate"

    r.run("preprocess(): centered, gray bars, RGB, 0..1", t_layout)
    r.run("as_rows(): values-first -> one row per candidate", t_rows)


def _a_szurp_v8(r, m):
    inp = Tensor("images", [1, 3, 480, 640], np.dtype(np.float32))

    def t_camera():
        x, tf = m.preprocess(_frame(1080, 1440), inp)
        _check_tensor(x, tf, (1, 3, 480, 640), np.float32)
        assert tf.pad_x == 0 and tf.pad_y == 0 and math.isclose(tf.scale, 480 / 1080), \
            "a 1440x1080 frame has the input's 4:3 shape: it should just be resized, with no padding"
        assert x.max() <= 1.0 + 1e-6, "values should be 0..1"
        assert math.isclose(float(x[0, 2, 200:300].mean()), 1.0), "RGB: pure blue should end up in channel 2"

    def t_pad():
        x, tf = m.preprocess(_frame(480, 1280), inp)
        assert tf.pad_x == 0 and tf.pad_y == 120 and math.isclose(tf.scale, 0.5), \
            f"a 1280x480 frame should be centered with 120 rows of padding above and below, got pad ({tf.pad_x}, {tf.pad_y})"
        assert _close(x[0, :, :120].mean(), 124 / 255), "RobotPilots pad with gray 124 (not 114 like Ultralytics)"

    def t_rows():
        a = np.random.default_rng(0).random((1, 21, 6300)).astype(np.float32)
        got = m.as_rows(a)
        assert got.shape == (6300, 21) and _close(got[9], a[0, :, 9]), "as_rows() should give (6300, 21): one row per guess"

    r.run("preprocess(): 4:3 camera frame, just resized", t_camera)
    r.run("preprocess(): other shapes centered on gray 124", t_pad)
    r.run("as_rows(): one column per guess -> one row per guess", t_rows)


def _a_end2end(r, m):
    def t_layout():
        x, tf = m.preprocess(_frame(480, 1280), Tensor("images", [1, 3, 576, 768], np.dtype(np.float32)))
        _check_tensor(x, tf, (1, 3, 576, 768), np.float32)
        assert math.isclose(tf.scale, 0.6) and tf.pad_y == 144 and tf.pad_x == 0, \
            "576 high x 768 wide: a 1280x480 frame scales by 0.6 and gets 144 rows of bars on top and bottom"
        assert math.isclose(float(x[0, 2, 300].mean()), 1.0), "RGB: pure blue should end up in channel 2"
        assert _close(x[0, :, :144].mean(), 114 / 255), "the bars should be gray 114, values 0..1"

    def t_camera():
        _, tf = m.preprocess(_frame(1080, 1440), Tensor("images", [1, 3, 576, 768], np.dtype(np.float32)))
        assert tf.pad_x == 0 and tf.pad_y == 0, "a 1440x1080 camera frame fits 768x576 exactly: no bars at all"

    def t_class():
        for cid, want in [(0, ("sentry", "small")), (3, ("4", "small")), (7, ("1", "big")), (11, ("base", "big"))]:
            got = tuple(m.class_info(cid))
            assert got == want, f"class_info({cid}) should be {want}, got {got}"

    r.run("preprocess(): non-square input, centered, gray bars, RGB, 0..1", t_layout)
    r.run("preprocess(): 4:3 camera frame needs no padding", t_camera)
    r.run("class_info(): s<size>_o<object> names", t_class)


def _a_yolo26(r, m):
    def t_layout():
        x, tf = m.preprocess(_frame(540, 720), Tensor("images", [1, 3, 640, 640], np.dtype(np.float32)))
        _check_tensor(x, tf, (1, 3, 640, 640), np.float32)
        assert tf.pad_x == 0 and tf.pad_y == 80, f"a 720x540 frame should be centered (80 rows of padding on top), got pad ({tf.pad_x}, {tf.pad_y})"
        assert math.isclose(float(x[0, 2, 200:400].mean()), 1.0), "RGB: pure blue should end up in channel 2"
        assert _close(x[0, :, :80].mean(), 114 / 255), "the padding should be gray 114, values 0..1"

    def t_labels():
        for cid, want in [(0, ("red", "sentry")), (3, ("red", "3")), (5, ("blue", "1")), (7, ("blue", "3"))]:
            got = tuple(m.label_info(cid))
            assert got == want, f"label_info({cid}) should be {want}, got {got}"

    r.run("preprocess(): centered, gray 114, RGB, 0..1", t_layout)
    r.run("label_info(): color and number from names like blue_1", t_labels)


# ============================================================================ Part B (decode)
def check_part_b(family, module):
    r = _Report(f"{family}: decode")
    {"yolov5": _b_yolov5, "yolox": _b_yolox, "ultralytics": _b_ultralytics, "szurp_v8": _b_szurp_v8,
     "end2end": _b_end2end, "yolo26_e2e": _b_yolo26}[family](r, module)
    return r.done()


def _logit(p):
    return math.log(p / (1 - p))


def _b_yolov5(r, m):
    tf = Transform(0.5, 0, 0, 640, 640)
    out = np.full((1, 25200, 22), -6.0, np.float32)

    def row(corners_tl_bl_br_tr, score, color, number):
        v = np.full(22, -6.0, np.float32)
        v[:8] = np.asarray(corners_tl_bl_br_tr, np.float32).reshape(-1)
        v[8], v[9 + color], v[13 + number] = _logit(score), 5.0, 5.0
        return v

    a = [(100, 100), (100, 140), (220, 142), (220, 98)]
    out[0, 100] = row(a, 0.95, 0, 1)
    out[0, 101] = row([(x + 3, y + 1) for x, y in a], 0.90, 0, 1)  # duplicate of a
    out[0, 5000] = row([(400, 300), (400, 330), (480, 331), (480, 299)], 0.80, 1, 6)
    out[0, 7000] = row([(50, 500), (50, 530), (120, 531), (120, 499)], 0.50, 1, 2)  # below threshold

    def t_decode():
        dets = m.decode([out], tf)
        assert isinstance(dets, list), "decode() should return a list of detection dicts"
        assert len(dets) == 2, f"expected 2 plates (one under the threshold, one duplicate), got {len(dets)}"
        dets = sorted(dets, key=lambda d: -d["score"])
        assert math.isclose(dets[0]["score"], 0.95, abs_tol=1e-3), "scores should be probabilities (sigmoid of column 8)"
        assert (dets[0]["color"], dets[0]["number"]) == ("blue", "1"), "first plate should be blue 1"
        assert (dets[1]["color"], dets[1]["number"]) == ("red", "outpost"), "second plate should be red outpost"
        want = np.array([a[0], a[3], a[2], a[1]], np.float32) / 0.5
        assert _close(_corners(dets), want), f"corners should be TL, TR, BR, BL in frame pixels: {want.tolist()}"

    r.run("decode(): threshold, labels, corner order, NMS", t_decode)


def _b_yolox(r, m):
    tf = Transform(0.5, 0, 52, 416, 416)
    rows = np.zeros((3549, 21), np.float32)
    cell = 2704 + 7 * 26 + 5  # stride 16, x 5, y 7
    rows[cell, :8] = [-1, -0.5, -1, 0.5, 1, 0.5, 1, -0.5]  # TL, BL, BR, TR in cells
    rows[cell, 8], rows[cell, 9 + 1], rows[cell, 13 + 3] = 0.9, 0.95, 0.9
    rows[cell + 1] = rows[cell]
    rows[cell + 1, :8] -= [1, 0] * 4  # the next cell sees the same plate
    rows[cell + 1, 8] = 0.85
    rows[100, :8], rows[100, 8], rows[100, 9] = 1, 0.5, 0.9  # below threshold

    def t_decode():
        for name, out in [("(1, 3549, 21)", rows[None]), ("pnnx (1, 21, 3549)", rows.T[None])]:
            dets = m.decode([out], tf)
            assert len(dets) == 1, f"{name}: expected 1 plate after threshold + NMS, got {len(dets)}"
            d = dets[0]
            assert math.isclose(d["score"], 0.9, abs_tol=1e-3), f"{name}: column 8 is already a probability"
            assert (d["color"], d["number"]) == ("red", "3"), f"{name}: expected red 3, got {d['color']} {d['number']}"
            in_px = np.array([[4, 6.5], [6, 6.5], [6, 7.5], [4, 7.5]]) * 16  # (cell + offset) * stride, TL TR BR BL
            want = (in_px - [0, 52]) / 0.5
            assert _close(_corners(dets), want, 1e-2), \
                f"{name}: corners should be TL, TR, BR, BL in frame pixels (undo scale AND padding): {want.tolist()}"

    r.run("decode(): grid decode, threshold, labels, both layouts", t_decode)


def _b_ultralytics(r, m):
    def t_yolo11():
        tf = Transform(640 / 720, 0, 80, 640, 640)
        out = np.zeros((1, 50, 8400), np.float32)
        kpts = np.array([[300, 300], [300, 340], [400, 342], [400, 298]], np.float32)  # TL, BL, BR, TR
        for j, score in [(5000, 0.9), (5001, 0.85), (100, 0.5)]:
            out[0, :4, j] = [350, 320, 100, 44]
            out[0, 4 + 12, j] = score  # class 12 = "Bfour"
            out[0, 42:50, j] = kpts.reshape(-1) + (j - 5000)
        dets = m.decode([out], tf)
        assert len(dets) == 1, f"expected 1 plate (one below threshold, one duplicate), got {len(dets)}"
        d = dets[0]
        assert math.isclose(d["score"], 0.9, abs_tol=1e-3), "the score is the best class score"
        assert (d["color"], d["number"], d["size"]) == ("blue", "4", "small"), \
            f"class 12 of yolo11 is 'Bfour': blue 4 small, got {d['color']} {d['number']} {d['size']}"
        want = (kpts[[0, 3, 2, 1]] - [0, 80]) / (640 / 720)
        assert _close(_corners(dets), want, 1e-2), f"corners should be TL, TR, BR, BL in frame pixels: {want.tolist()}"

    def t_yolov8():
        tf = Transform(1.0, 0, 0, 416, 416)
        out = np.zeros((1, 14, 3549), np.float32)
        out[0, :4, 7] = [50, 50, 40, 20]
        out[0, 5, 7] = 0.8  # class 1 = "R"
        out[0, 6:14, 7] = [30, 40, 30, 60, 70, 60, 70, 40]
        dets = m.decode([out], tf)
        assert len(dets) == 1 and dets[0]["color"] == "red", "reborn_yolov8 has 2 classes: B and R"
        assert dets[0]["number"] == "?", "reborn_yolov8 doesn't know the number: use '?'"

    r.run("decode(): tongji_yolo11 (38 classes)", t_yolo11)
    r.run("decode(): reborn_yolov8 (2 classes, color only)", t_yolov8)


def _b_szurp_v8(r, m):
    tf = Transform(0.5, 0, 120, 480, 640)
    out = np.zeros((1, 21, 6300), np.float32)
    kpts = np.array([[300, 200], [300, 240], [400, 242], [400, 198]], np.float32)  # TL, BL, BR, TR
    for j, score, shift in [(1000, 0.8, 0), (1001, 0.7, 2), (2000, 0.4, 50)]:
        out[0, 0:4, j] = [0.05, 0.9, 0.02, 0.0]   # red
        out[0, 4 + 7, j] = score                  # class 7: base (bottom), a big plate
        out[0, 13:21, j] = (kpts + shift).reshape(-1)

    def t_decode():
        dets = m.decode([out], tf)
        assert len(dets) == 1, f"expected 1 plate (one duplicate, one below the 0.5 threshold), got {len(dets)}"
        d = dets[0]
        assert math.isclose(d["score"], 0.8, abs_tol=1e-3), "the score is the best of the 9 class scores"
        assert (d["color"], d["number"], d["size"]) == ("red", "base", "big"), \
            f"columns 0-3 are colors (red wins), class 7 is the base's big bottom plate: got {d['color']} {d['number']} {d['size']}"
        want = (kpts[[0, 3, 2, 1]] - [0, 120]) / 0.5
        assert _close(_corners(dets), want, 1e-2), f"corners (columns 13-20) should be TL, TR, BR, BL in frame pixels: {want.tolist()}"

    r.run("decode(): colors first, class score, corners last, NMS", t_decode)


def _b_end2end(r, m):
    tf = Transform(0.5, 0, 52, 576, 768)
    out = np.zeros((1, 30, 18), np.float32)
    kpts = np.array([[100, 152], [100, 172], [160, 174], [160, 150]], np.float32)  # TL, BL, BR, TR
    out[0, 0, :6] = [100, 150, 160, 174, 0.92, 7.0]  # class 7: s1_o1 = big 1
    out[0, 0, 6:10] = [0, 1, 0, 0]
    out[0, 0, 10:18] = kpts.reshape(-1)
    out[0, 1, :6] = [300, 200, 340, 220, 0.3, 2.0]   # below threshold
    out[0, 2, :6] = [400, 200, 440, 220, 0.8, 3.0]   # class 3: s0_o4 = small 4
    out[0, 2, 6:10] = [1, 0, 0, 0]
    out[0, 2, 10:18] = [400, 200, 400, 220, 440, 220, 440, 200]

    def t_decode():
        dets = m.decode([out], tf)
        assert len(dets) == 2, f"expected 2 plates (one row is below the threshold), got {len(dets)}"
        a, b = dets
        assert a["score"] >= b["score"], "sort the plates by score, best first"
        assert (a["color"], a["number"], a["size"]) == ("red", "1", "big"), \
            f"row 0 is class 7 (s1_o1) with color 1: red 1 big, got {a['color']} {a['number']} {a['size']}"
        assert (b["color"], b["number"], b["size"]) == ("blue", "4", "small"), "row 2 should be blue 4 small"
        want = (kpts[[0, 3, 2, 1]] - [0, 52]) / 0.5
        assert _close(_corners(dets), want, 1e-2), f"corners should be TL, TR, BR, BL in frame pixels: {want.tolist()}"

    r.run("decode(): threshold, class + color, corner order (no NMS needed)", t_decode)


def _b_yolo26(r, m):
    tf = Transform(0.5, 0, 80, 640, 640)
    out = np.zeros((1, 300, 18), np.float32)
    kpts = np.array([[100, 152], [100, 172], [160, 174], [160, 150]], np.float32)  # TL, BL, BR, TR
    out[0, 0, :6] = [100, 150, 160, 174, 0.92, 5.0]  # class 5: blue_1
    out[0, 0, 6:18] = np.hstack([kpts, np.full((4, 1), 0.99)]).reshape(-1)  # x, y, visible
    out[0, 1, :6] = [300, 200, 340, 220, 0.2, 2.0]   # below the threshold
    out[0, 2, :6] = [400, 200, 440, 220, 0.7, 3.0]   # class 3: red_3
    out[0, 2, 6:18] = [400, 200, 0.9, 400, 220, 0.9, 440, 220, 0.9, 440, 200, 0.9]

    def t_decode():
        dets = m.decode([out], tf)
        assert len(dets) == 2, f"expected 2 plates (one row is below the threshold), got {len(dets)}"
        a, b = dets
        assert a["score"] >= b["score"], "sort the plates by score, best first"
        assert (a["color"], a["number"]) == ("blue", "1") and (b["color"], b["number"]) == ("red", "3"), \
            f"rows 0 and 2 are classes 5 (blue_1) and 3 (red_3), got {a['color']} {a['number']} and {b['color']} {b['number']}"
        want = (kpts[[0, 3, 2, 1]] - [0, 80]) / 0.5
        assert _close(_corners(dets), want, 1e-2), \
            f"corners are (x, y, visible) triples: keep x and y, order TL, TR, BR, BL, frame pixels: {want.tolist()}"

    r.run("decode(): score, class, (x, y, visible) corners (no NMS needed)", t_decode)


# ============================================================================ m1, m6
def check_m1(onnx_card, openvino_card):
    r = _Report("m1: reading IO from the files")

    def need(name):
        m = zoo.BY_NAME[name]
        if not m.exists():
            raise AssertionError(f"{m.file} is missing (is MODELS_DIR right?)")
        return m.file

    def t_onnx():
        c = onnx_card(need("rp_0526"))
        assert set(c) >= {"inputs", "outputs", "producer", "ops", "metadata"}, \
            'return a dict with "inputs", "outputs", "producer", "ops" and "metadata"'
        assert [tuple(map(str, (n, list(s), d))) for n, s, d in c["inputs"]] == [("images", "[1, 3, 640, 640]", "float16")], \
            f"rp_0526's input should be ('images', [1, 3, 640, 640], 'float16'), got {c['inputs']}"
        assert [(n, list(s)) for n, s, _ in c["outputs"]] == [("output", [1, 25200, 22])], \
            f"rp_0526's output should be ('output', [1, 25200, 22], ...), got {c['outputs']}"
        assert "pytorch" in c["producer"].lower(), "producer should say pytorch (producer_name + producer_version)"
        assert c["ops"].get("Conv", 0) > 50, "ops should count every node's op_type"

    def t_onnx_meta():
        c = onnx_card(need("tongji_yolo11"))
        assert "Bfour" in c["metadata"].get("names", ""), "Ultralytics stores the class names in metadata_props['names']"

    def t_openvino():
        c = openvino_card(need("tongji_yolo11_int8"))
        assert [(n, list(s)) for n, s, _ in c["inputs"]] == [("x", [1, 3, 640, 640])], \
            f"tongji_yolo11_int8.xml's input is called 'x', got {c['inputs']}"
        (name, shape, dtype), = c["outputs"]
        assert name is None and list(shape) == [1, 50, 8400], \
            f"tongji_yolo11_int8.xml has one output (1, 50, 8400) with NO name (use None), got {c['outputs']}"
        assert "Bfour" in str(c["metadata"].get("model_info/labels", "")), \
            "read the labels from rt_info: model_info/labels"

    def t_openvino_source():
        c = openvino_card(need("rp_0708_openvino"))
        assert "0708.onnx" in str(c["metadata"].get("conversion_parameters/input_model", "")), \
            "rp_0708_openvino.xml's rt_info records which file it was converted from (conversion_parameters/input_model)"

    r.run("onnx_card(): rp_0526 inputs, outputs, producer, ops", t_onnx)
    r.run("onnx_card(): metadata (tongji_yolo11 labels)", t_onnx_meta)
    r.run("openvino_card(): tongji_yolo11_int8.xml, including its nameless output", t_openvino)
    r.run("openvino_card(): rt_info says where rp_0708_openvino.xml came from", t_openvino_source)
    return r.done()


def check_m6(compare):
    r = _Report("m6: comparing two files' detections")

    def det(x, y, color="blue", number="3", score=0.9):
        c = np.array([[x - 20, y - 10], [x + 20, y - 10], [x + 20, y + 10], [x - 20, y + 10]], np.float32)
        return {"corners": c, "score": score, "color": color, "number": number, "size": None, "label": ""}

    def t_compare():
        a = [det(100, 100), det(300, 300, score=0.8)]
        b = [det(101, 100, score=0.88), det(300, 302, number="4", score=0.8), det(600, 600)]
        c = compare(a, b)
        assert c["matched"] == 2 and c["only_a"] == 0 and c["only_b"] == 1, \
            f"2 plates match, B found 1 extra: got matched {c['matched']}, only_a {c['only_a']}, only_b {c['only_b']}"
        assert math.isclose(c["corner_px"], 2.0, abs_tol=1e-3), f"worst corner distance is 2 px, got {c['corner_px']}"
        assert math.isclose(c["score_diff"], 0.02, abs_tol=1e-3), f"worst score difference is 0.02, got {c['score_diff']}"
        assert c["labels_agree"] is False, "one pair has number 3 vs 4: labels_agree should be False"

    def t_empty():
        c = compare([], [det(5, 5)])
        assert c["matched"] == 0 and c["only_b"] == 1 and c["corner_px"] is None and c["score_diff"] is None, \
            "with nothing matched, corner_px and score_diff should be None"

    r.run("compare(): matches, extras, worst corner and score difference", t_compare)
    r.run("compare(): nothing matched", t_empty)
    return r.done()
