"""Exercise 2: load the TensorRT engine you built and run it from Python.

Build the engine first (see README, exercise 2), then run:  python3 ex2_load_engine.py
"""
import time

import numpy as np
import tensorrt as trt
import torch

import checks
import config

# TensorRT data types -> torch data types (torch is only used to own the GPU memory)
TRT_TO_TORCH = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.int8: torch.int8,
    trt.bool: torch.bool,
}


class TRTModel:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        # The engine contains the EfficientNMS_TRT plugin; it must be registered before loading
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Could not load {engine_path}. Was it built on this device and TensorRT version?")
        self.context = self.engine.create_execution_context()

        # One GPU buffer per input/output tensor; tell TensorRT where each one lives
        self.buffers, self.inputs, self.outputs = {}, [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]
            self.buffers[name] = torch.empty(shape, dtype=dtype, device="cuda")
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            (self.inputs if is_input else self.outputs).append(name)
        self.stream = torch.cuda.current_stream().cuda_stream

    def io_info(self):
        """List of (name, "INPUT"/"OUTPUT", shape, dtype) for every tensor."""
        return [(n, self.engine.get_tensor_mode(n).name, tuple(self.buffers[n].shape), str(self.buffers[n].dtype))
                for n in self.inputs + self.outputs]

    def __call__(self, tensor):
        """tensor: (1, 3, 640, 640) float32 numpy array. Returns {output_name: numpy array}."""
        inp = self.buffers[self.inputs[0]]
        inp.copy_(torch.from_numpy(np.ascontiguousarray(tensor)).to(inp.dtype))  # CPU -> GPU
        self.context.execute_async_v3(self.stream)
        torch.cuda.synchronize()  # wait for the GPU to finish
        return {n: self.buffers[n].cpu().numpy() for n in self.outputs}  # GPU -> CPU


if __name__ == "__main__":
    if not checks.check_ex2(TRTModel):
        raise SystemExit
    model = TRTModel(config.ENGINE_PATH)
    for name, mode, shape, dtype in model.io_info():
        print(f"{mode:6s} {name:22s} {shape}  {dtype}")

    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    model(x)  # warm-up
    t0 = time.perf_counter()
    for _ in range(100):
        model(x)
    print(f"\n{(time.perf_counter() - t0) * 10:.2f} ms per inference, including copies to and from the GPU")
