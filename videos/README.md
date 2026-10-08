# Videos

- `demo.avi`: the footage both tracks use by default. 1440x1080 MJPEG, 30 fps, 687 frames (23 s), recorded with a
  HikRobot camera (its settings and calibration are in `configs/demo_camera.yaml`, the same numbers as in
  `exercises/config.py` and `model_zoo/zoo.py`). Every plate in it is **blue**. Same bytes as
  `assets/demo/demo.avi` in our [sp_vision fork](https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25).

**Your own recordings go here too**, with their labels next to them:

```
python3 shared/cameras.py --source hik --record videos/ours1.avi --seconds 30
cd model_zoo && python3 label_frames.py --source ../videos/ours1.avi --out ../videos/ours1_labels.json
```

Record red AND blue plates, near and far, moving and spinning, in match lighting. MJPEG at 1440x1080 is about
2.7 MB per second and GitHub refuses files over 100 MB, so keep clips to 30 seconds or less.
