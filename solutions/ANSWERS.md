# Answers to the README questions

Short reference answers for whoever reviews the exercises. Good answers from members will often say more.
Words in these answers are explained in [BASICS.md](../BASICS.md).

## Exercise 1
1. `images`, shape `[1, 3, 640, 640]`: 1 picture at a time, 3 color channels, 640 rows, 640 columns.
2. `output/sink_port_0` (1×25200×22) is the network's own answer. `num_detections`, `detection_boxes`,
   `detection_scores` and `detection_classes` were added later, by a script in the robot's repo
   (`scripts/onnx/fuse_efficient_nms.py`) that bolted an NMS step onto the end.
3. YOLOv5 guesses on three grids: 80×80, 40×40 and 20×20 cells for a 640×640 input (strides 8, 16, 32), with
   3 guesses per cell (one per anchor): (6400 + 1600 + 400) × 3 = 25200.
4. NVIDIA's NMS step for TensorRT (a "plugin": an add-on operation). Only TensorRT can run it. That's why the
   laptop mode uses the copy of the model without it (`rp_0708_openvino2onnx.onnx`).
5. The box input is `Concat(min_y, min_x, max_y, max_x)`: the boxes are **y first**. Drawing them as
   x1, y1, x2, y2 swaps the two directions: move a plate left-right and the box moves up-down.
6. RobotPilots trained the model and published it as an ONNX file (`models/robotpilots/rp_0708.onnx`). Tongji
   converted that to OpenVINO's format and only shipped the OpenVINO version in their robot code. There's no
   official way back from OpenVINO to ONNX, so our team used the `openvino2onnx` tool to get an ONNX file
   TensorRT could use. (We only found the original 0708.onnx later: the model zoo compares all of them.)

## Exercise 2
1. FP16 is usually close to twice as fast on the Orin with almost no change in the answers: a model's numbers
   rarely need more digits than FP16 keeps.
2. An engine contains GPU code chosen and tuned for one exact GPU and one TensorRT/CUDA version. Unlike an ONNX
   file it isn't portable: rebuild it on each device.
3. Loading fails, because the engine uses `EfficientNMS_TRT`, which TensorRT can't find until
   `trt.init_libnvinfer_plugins` registers it.
4. Usually slower than trtexec's GPU time: the Python timing also copies the input to the GPU and every output
   back, waits for the GPU, and runs Python code.

## Exercise 3
1. Stretching changes the plates' shape, and the model learned on unstretched pictures, so it gets worse.
2. Top-left (`cv::Rect(0, 0, w, h)`), with the rest black. Many examples center the picture and fill with gray
   (114). The model was trained one way, and decoding has to undo that same way: get it wrong and every corner
   is shifted.
3. BGR → RGB, and whole numbers 0–255 → decimals 0–1. Also (height, width, channels) → (channels, height,
   width), plus a batch dimension in front.
4. To turn the model's 640×640 coordinates back into frame pixels (divide by scale), in exercises 4–7.

## Exercise 4
1. Columns 0–7 are 4 (x, y) corners in the order top-left, bottom-left, bottom-right, top-right
   (counter-clockwise on screen). The robot's code reorders them to top-left, top-right, bottom-right,
   bottom-left.
2. It's a logit, a raw score. `sigmoid(x) = 1 / (1 + e^-x)` turns it into 0..1.
3. 9–12 are the color (blue, red, gray = switched off, purple); 13–21 are the plate's number (sentry, 1–5,
   outpost, base, not armor). In each group, take the index of the biggest score (argmax).
4. The 4 corners show the plate's shape in the picture, which tells you how it's turned, so solvePnP can work out
   its 3D position and which way it faces. A plain rectangle can't tell a turned plate from a small one.

## Exercise 5
1. Neighboring grid cells and their anchors all see the same plate, so each of them reports it.
2. NMS keeps the best-scoring guess and removes others that overlap it by more than the IoU threshold. A lower
   threshold removes more, but may merge two real plates that are close together.
3. Sigmoid always goes up as its input goes up, so `sigmoid(x) >= t` exactly when `x >= log(t / (1 - t))`.
   Comparing logits skips 25200 sigmoid calculations per frame; only the few rows that pass get one.
4. Typically several milliseconds for a Python loop, and well under one millisecond for numpy.

## Exercise 6
1. A smaller area means each plate covers more of the model's input (better at long range) and there's less
   clutter. The risk is missing anything outside the ROI.
2. Yes, until the crop is smaller than 640×640: past that the model sees an enlarged picture with no extra detail.
   The camera's resolution sets the real limit.
3. Whichever is lower. With TensorRT FP16, detection usually beats a 30–60 FPS camera, so the camera (or the
   decoding/drawing code) is the limit.
4. Bonus: the boxes are (min_y, min_x, max_y, max_x) in 640×640 pixels. Swap to x first, divide by scale, and
   add the ROI's top-left corner.

## Exercise 7
1. The real plate's corners in 3D (its size), the camera matrix (focal length and center) and the distortion
   coefficients. Without the real size the distance can't be known; without the calibration, pixels can't be
   turned into directions.
2. solvePnP pairs 3D point number i with picture point number i. Shuffled points give a wrong position, or a
   mirrored one.
3. They describe one specific camera and lens (focal length in pixels, center, lens bending). Another camera,
   lens or picture size has different values, so each camera gets calibrated.
4. The distance grows with the size you assume. Using a 230 mm plate when it's really 135 mm makes it look about
   1.7× farther away.
