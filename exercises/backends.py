"""Picks how the model runs: your TensorRT engine (Jetson) or ONNX Runtime (any laptop).

Both return the same thing: a callable that takes a (1, 3, 640, 640) float32 array
and returns a dict of {output_name: numpy array}.
"""
import numpy as np

import config


class OrtModel:
    """CPU fallback. Uses the ONNX file WITHOUT the fused NMS, because ONNX Runtime
    can't run TensorRT's EfficientNMS_TRT plugin. You only get the raw output."""

    def __init__(self, onnx_path):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def __call__(self, tensor):
        outs = self.session.run(None, {"images": np.ascontiguousarray(tensor, dtype=np.float32)})
        return dict(zip(self.output_names, outs))


def get_model():
    if config.BACKEND == "onnxruntime":
        return OrtModel(config.ONNX_NO_NMS_PATH)
    try:
        from ex2_load_engine import TRTModel
    except ImportError as e:
        raise SystemExit(f"Can't use TensorRT here ({e}). On a laptop, run with BACKEND=onnxruntime.")

    if not config.ENGINE_PATH.exists():
        raise SystemExit(f"No engine at {config.ENGINE_PATH}. Build it first (exercise 2), "
                         "or set BACKEND=onnxruntime to run on the CPU.")
    try:
        return TRTModel(config.ENGINE_PATH)
    except NotImplementedError:
        raise SystemExit("Finish exercise 2 (TRTModel) first, or set BACKEND=onnxruntime to run on the CPU.")


def read_frame(source=None, index=60):
    """Grab frame number `index` from a video (default: the demo video) or any camera cameras.py opens."""
    from cameras import open_camera

    try:
        with open_camera(source or config.VIDEO_PATH) as cam:
            for _, frame in cam.frames(start=index, count=1):
                return frame
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"Could not open {source or config.VIDEO_PATH}: {e}") from None
    raise SystemExit(f"Could not read frame {index} from {source or config.VIDEO_PATH}")
