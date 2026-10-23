"""Load any model file and run it, whatever its format.

    .onnx   -> ONNX Runtime  (any laptop or the Jetson's CPU; can't run TensorRT-only nodes)
    .xml    -> OpenVINO      (any laptop or the Jetson's CPU; its GPU support is Intel-only)
    .param  -> NCNN          (CPU)
    .engine -> TensorRT      (NVIDIA GPU; only on the device + TensorRT version that built it)

Every runner looks the same from the outside, so the rest of the zoo doesn't care about formats:
    runner.inputs, runner.outputs  -> lists of Tensor(name, shape, dtype)
    runner(x)                      -> list of numpy arrays, in the same order as runner.outputs
    runner.metadata                -> dict of what the file says about itself (labels, sizes, ...)

Runners never fix your input for you. Pass float32 to a float16 model and you get an error
that says so: building the right input is your job (that's what m3-m6 are about).
"""
import os
from collections import namedtuple
from pathlib import Path

import numpy as np

Tensor = namedtuple("Tensor", "name shape dtype")


def load(model, device=None):
    """model: a zoo.ModelFile or a path. Returns the runner for its format."""
    path = Path(getattr(model, "file", model))
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist")
    if path.suffix == ".onnx":
        return OnnxRunner(path)
    if path.suffix == ".xml":
        return OpenVinoRunner(path, device)
    if path.suffix == ".param":
        return NcnnRunner(path, getattr(model, "input_hw", None))
    if path.suffix == ".engine":
        return TensorRTRunner(path)
    raise ValueError(f"Don't know how to run {path.name}")


def check_input(x, t):
    if not isinstance(x, np.ndarray):
        raise TypeError(f"input '{t.name}' must be a numpy array, got {type(x).__name__}")
    if x.dtype != t.dtype:
        raise TypeError(f"input '{t.name}' must be {t.dtype}, got {x.dtype} (look at this model's IO card in m1)")
    fixed = [d if isinstance(d, int) and d > 0 else None for d in t.shape]
    if len(x.shape) != len(fixed) or any(f is not None and f != s for f, s in zip(fixed, x.shape)):
        raise ValueError(f"input '{t.name}' must have shape {tuple(t.shape)}, got {x.shape}")


def engine_trt_version(path):
    """Which TensorRT built this engine. Read from the file header: bytes 24-27 have held
    major.minor.patch.build on every engine we've seen. NVIDIA doesn't document it, so it's a hint."""
    with open(path, "rb") as f:
        head = f.read(28)
    return ".".join(str(b) for b in head[24:28]) if head[:4] == b"ftrt" and len(head) == 28 else None


# ----------------------------------------------------------------------------- ONNX Runtime
ORT_TYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
             "tensor(int64)": np.int64, "tensor(int32)": np.int32, "tensor(uint8)": np.uint8,
             "tensor(bool)": np.bool_}


class OnnxRunner:
    backend = "onnxruntime"

    def __init__(self, path):
        import onnxruntime as ort

        # CPU by default. On a Jetson with onnxruntime-gpu: ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
        providers = os.environ.get("ORT_PROVIDERS", "CPUExecutionProvider").split(",")
        options = ort.SessionOptions()
        # By default ORT's threads keep spinning after a run, stealing the CPU from whatever runs
        # next (another model, OpenVINO, ...). That makes side-by-side timings meaningless.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        try:
            try:
                self.session = ort.InferenceSession(str(path), options, providers=providers)
            except Exception as e:
                # On ARM (the Jetson), ORT's top optimization level rewrites float16 ops into its own
                # NHWC layout, then finds it has no kernel for some of them (rp_0526's MaxPool).
                # One level down skips that layout rewrite; the model itself is plain ONNX.
                if "com.ms.internal.nhwc" not in str(e):
                    raise
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
                self.session = ort.InferenceSession(str(path), options, providers=providers)
        except Exception as e:  # onnxruntime raises its own exception types
            if "EfficientNMS_TRT" in str(e):
                raise RuntimeError(f"{path.name} contains EfficientNMS_TRT, a TensorRT plugin that ONNX Runtime "
                                   "can't run. Build an engine from it on the Jetson, or run the same network "
                                   "without the NMS node (rp_0708_openvino2onnx, or rp_0708).") from None
            raise
        self.inputs = [self._tensor(t) for t in self.session.get_inputs()]
        self.outputs = [self._tensor(t) for t in self.session.get_outputs()]
        meta = self.session.get_modelmeta()
        self.metadata = {"producer": meta.producer_name, **meta.custom_metadata_map}

    @staticmethod
    def _tensor(t):
        return Tensor(t.name, list(t.shape), np.dtype(ORT_TYPES.get(t.type, np.float32)))

    def __call__(self, x):
        check_input(x, self.inputs[0])
        return self.session.run(None, {self.inputs[0].name: x})


# ----------------------------------------------------------------------------- OpenVINO
class OpenVinoRunner:
    backend = "openvino"
    # rt_info is OpenVINO's metadata. These are the entries worth reading on our models.
    RT_INFO = [("model_info", "labels"), ("model_info", "pad_value"), ("model_info", "reverse_input_channels"),
               ("model_info", "scale_values"), ("conversion_parameters", "input_model"),
               ("conversion_parameters", "framework"), ("framework", "description"), ("framework", "names"),
               ("framework", "imgsz"), ("framework", "kpt_shape"), ("Runtime_version",)]

    def __init__(self, path, device=None):
        import openvino as ov

        core = ov.Core()
        model = core.read_model(str(path))
        self.metadata = {"/".join(k): model.get_rt_info(list(k)).astype(str)
                         for k in self.RT_INFO if model.has_rt_info(list(k))}
        if model.has_rt_info(["nncf", "quantization"]):
            self.metadata["nncf"] = "quantized with NNCF (INT8 weights and activations)"
        self.inputs = [self._tensor(p) for p in model.inputs]
        self.outputs = [self._tensor(p) for p in model.outputs]
        self.device = device or os.environ.get("OV_DEVICE", "CPU")
        self.compiled = core.compile_model(model, self.device)
        self.request = self.compiled.create_infer_request()

    @staticmethod
    def _tensor(port):
        names = sorted(port.get_names())  # can be empty: yolo11.xml's output has no name at all
        shape = [d.get_length() if d.is_static else None for d in port.get_partial_shape()]
        return Tensor(names[0] if names else None, shape, np.dtype(port.get_element_type().to_dtype()))

    def __call__(self, x):
        check_input(x, self.inputs[0])
        result = self.request.infer({0: x})
        return [np.array(result[port]) for port in self.compiled.outputs]  # copy: the next infer reuses the buffer


# ----------------------------------------------------------------------------- NCNN
def _ncnn_blobs(param_path):
    """Input and output blob names from a .param file (it's plain text: one layer per line)."""
    inputs, produced, consumed = [], [], set()
    for line in Path(param_path).read_text().splitlines()[2:]:
        parts = line.split()
        if len(parts) < 4:
            continue
        kind, n_in, n_out = parts[0], int(parts[2]), int(parts[3])
        ins, outs = parts[4:4 + n_in], parts[4 + n_in:4 + n_in + n_out]
        consumed.update(ins)
        produced += outs
        if kind == "Input":
            inputs += outs
    return inputs, [b for b in produced if b not in consumed]


class NcnnRunner:
    backend = "ncnn"

    def __init__(self, param_path, input_hw=None):
        import ncnn

        param_path = Path(param_path)
        in_names, out_names = _ncnn_blobs(param_path)
        # A .param file doesn't record the input size. You have to know it (zoo.py: input_hw).
        hw = list(input_hw) if input_hw else [None, None]
        self.inputs = [Tensor(n, [1, 3] + hw, np.dtype(np.float32)) for n in in_names]
        self.outputs = [Tensor(n, None, np.dtype(np.float32)) for n in out_names]
        self.metadata = {"converter": "pnnx" if param_path.name.endswith(".ncnn.param") else "onnx2ncnn",
                         "weights": param_path.with_suffix(".bin").name}
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        if self.net.load_param(str(param_path)) or self.net.load_model(str(param_path.with_suffix(".bin"))):
            raise RuntimeError(f"NCNN couldn't load {param_path.name}")

    def __call__(self, x):
        import ncnn

        check_input(x, self.inputs[0])
        ex = self.net.create_extractor()
        ex.input(self.inputs[0].name, ncnn.Mat(np.ascontiguousarray(x[0])))  # NCNN has no batch dimension
        outs = []
        for t in self.outputs:
            ret, mat = ex.extract(t.name)
            if ret != 0:
                raise RuntimeError(f"NCNN couldn't compute blob {t.name}")
            outs.append(np.array(mat)[None])  # put the batch dimension back
        return outs


# ----------------------------------------------------------------------------- TensorRT
class TensorRTRunner:
    """Same steps as exercise 2's TRTModel: torch owns the GPU buffers."""
    backend = "tensorrt"

    def __init__(self, path):
        import tensorrt as trt
        import torch

        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")  # the course model's engine has the EfficientNMS_TRT plugin
        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        built = engine_trt_version(path)
        if self.engine is None:
            raise RuntimeError(f"TensorRT {trt.__version__} can't load {Path(path).name} (built with TensorRT "
                               f"{built or '?'}). Engines only load on the GPU + TensorRT version that built "
                               "them: rebuild it from the ONNX with jetson/build_engines.py.")
        self.context = self.engine.create_execution_context()
        to_torch = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
                    trt.int64: torch.int64, trt.int8: torch.int8, trt.bool: torch.bool}
        self.buffers, self.inputs, self.outputs = {}, [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = list(self.engine.get_tensor_shape(name))
            trt_dtype = self.engine.get_tensor_dtype(name)
            self.buffers[name] = torch.empty(shape, dtype=to_torch[trt_dtype], device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            t = Tensor(name, shape, np.dtype(trt.nptype(trt_dtype)))
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(t)
        # Our own CUDA stream: on the default stream TensorRT adds extra syncs (and prints a warning).
        self.torch_stream = torch.cuda.Stream()
        self.stream = self.torch_stream.cuda_stream
        self.metadata = {"tensorrt": trt.__version__, "built_with": built}

    def __call__(self, x):
        import torch

        check_input(x, self.inputs[0])
        with torch.cuda.stream(self.torch_stream):
            self.buffers[self.inputs[0].name].copy_(torch.from_numpy(np.ascontiguousarray(x)))  # CPU -> GPU
            self.context.execute_async_v3(self.stream)
            outs = [self.buffers[t.name].cpu() for t in self.outputs]  # GPU -> CPU, waits for the stream
        return [o.numpy() for o in outs]