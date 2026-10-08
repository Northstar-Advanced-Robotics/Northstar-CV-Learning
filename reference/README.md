# Reference: the robot's own code

The exercises sometimes say "look at how the robot does it". The robot's vision program lives in our fork of
Tongji SuperPower's `sp_vision_25`. You don't need to clone it: the files the exercises point to are copied here,
unchanged, from [Northstar-sp_vision_25](https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25)
(MIT license, see `sp_vision_25/LICENSE`).

**You don't need to know C++.** Each exercise tells you which function to read and what to look for. Most of the
time it's one or two lines, like a number or the order of four values.

| file | what's in it | used in |
|---|---|---|
| `sp_vision_25/tasks/auto_aim/yolos/yolov5.cpp` | how the robot prepares a camera frame for the model (`detect()`, and the `ppp` block near the top) and how it reads the model's answer (`parse()`) | exercises 3, 4, 5; zoo m2 |
| `sp_vision_25/tasks/auto_aim/yolos/yolov5.hpp` | the score threshold and other settings | exercises 4, 5; zoo m2 |
| `sp_vision_25/tasks/auto_aim/armor.cpp`, `armor.hpp` | what each color and number index means; the list of 38 classes (`armor_properties`) | exercise 4; zoo m4 |
| `sp_vision_25/tasks/auto_aim/solver.cpp` | the armor plate's real size and the `solvePnP` call that turns corners into a distance | exercise 7 |
| `sp_vision_25/tasks/auto_aim/yolos/yolo11.cpp`, `yolov8.cpp` | how the robot runs the other two models it supports | zoo m4 |
| `sp_vision_25/JETSON_ORIN.md` | where the robot's model files came from, and how it runs on the Jetson | exercises 1, 2; zoo MODELS.md |

Copied on 2026-10-08. If the robot's code changes, these copies don't: they're here to learn from, not to run.
