"""Exercise 2: load the TensorRT engine you built and run it from Python.

Build the engine first with trtexec (README, exercise 2), then fill in TRTModel and run:
    python3 ex2_load_engine.py

Words you'll meet (BASICS.md, section 10, explains engines):
    deserialize        turn the bytes of the .engine file back into an object you can use
    execution context  the object that actually runs the engine; you tell it which memory to use
    tensor             an input or output array, like the (1, 3, 640, 640) picture
    buffer             a block of memory ON THE GPU. The GPU can't read numpy arrays (they live in
                       the computer's normal memory), so every input and output needs a GPU buffer
    plugin             an extra operation TensorRT doesn't have built in, like EfficientNMS_TRT

Docs: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html
Use the TensorRT 10 functions (num_io_tensors, get_tensor_name, set_tensor_address,
execute_async_v3). Many examples online use an OLD way ("bindings") that no longer exists.
"""
import time

import numpy as np
import tensorrt as trt
import torch

import checks
import config

# We use PyTorch (torch) only to get GPU memory: torch.empty(shape, dtype=..., device="cuda") gives
# you a GPU buffer, .data_ptr() is its address (a number TensorRT needs), .copy_(x) copies data into
# it, and .cpu().numpy() copies it back out as a numpy array.
# TensorRT and torch name number types differently; this dict translates.
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
        # TODO
        # 1. Make a trt.Logger (where TensorRT prints its messages) and register TensorRT's built-in
        #    plugins with trt.init_libnvinfer_plugins(logger, ""). The engine contains EfficientNMS_TRT,
        #    which is a plugin: what happens if you skip this?
        # 2. Read the engine file's bytes (open(path, "rb").read()) and deserialize them with a
        #    trt.Runtime: runtime.deserialize_cuda_engine(data)
        # 3. Create an execution context: engine.create_execution_context()
        # 4. For every input and output tensor: look up its name, shape and dtype (number type),
        #    make a GPU buffer for it with torch, and tell the context that buffer's address.
        #    Remember which tensors are inputs and which are outputs (engine.get_tensor_mode).
        raise NotImplementedError

    def io_info(self):
        """Return a list of (name, "INPUT" or "OUTPUT", shape, dtype) for every tensor."""
        # TODO
        raise NotImplementedError

    def __call__(self, tensor):
        """tensor: (1, 3, 640, 640) float32 numpy array. Return {output_name: numpy array}."""
        # TODO: copy the input into its GPU buffer, run the engine, wait until the GPU is done,
        # then copy every output back into a numpy array
        raise NotImplementedError


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
