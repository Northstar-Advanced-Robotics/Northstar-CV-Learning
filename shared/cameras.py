"""Frames from any camera, video or stream, through one class. Used by both tracks: exercises 6-7
and every script in model_zoo/ take --source.

    python3 shared/cameras.py --list                           # which cameras can this machine see?
    python3 shared/cameras.py --source usb:0                   # live preview: r record, s snapshot, q quit
    python3 shared/cameras.py --source usb:0 --cam width=1280 --cam height=720 --cam fps=60
    python3 shared/cameras.py --source hik --cam exposure_ms=2 --record videos/match1.avi --seconds 60 --headless

Sources (the same strings work for --source in every script):
    demo.avi, match.mp4      a video file (the default is videos/demo.avi, or $VIDEO if you set it)
    shots/, "shots/*.png"    a folder or pattern of images
    usb:0, 0, /dev/video2    USB / UVC webcam through OpenCV (V4L2 on Linux, DirectShow on Windows)
    hik, hik:1               HikRobot industrial camera, MVS SDK (the robot's camera)
    mv, mv:0                 MindVision industrial camera, MindVision SDK
    csi:0                    Jetson CSI ribbon camera (GStreamer; needs OpenCV built with GStreamer)
    gst:<pipeline>           any GStreamer pipeline that ends in appsink
    rtsp://..., http://...   a network stream
    config:<file.yaml>       the camera a config file describes (camera_name: usb/hikrobot/mindvision),
                             e.g. configs/demo_camera.yaml. Same format as the robot's own config files

Settings: --cam KEY=VALUE, repeatable.
    usb:        width, height, fps, exposure, gain, gamma, fourcc (the camera driver's own units)
    hik:        exposure_ms, gain, fps          (defaults from the robot's configs: 2 ms, gain 16)
    mv:         exposure_ms, gamma
    csi:        width, height, fps
A camera can ignore what you ask for: the preview prints what it actually does.

Every camera hands out BGR uint8 frames, like cv2.imread. To add another kind of camera, subclass
Camera, implement _grab(), and register an opener with @register("name").

"sp_vision" in the comments below is the robot's vision code
(https://github.com/Northstar-Advanced-Robotics/Northstar-sp_vision_25): the camera settings here copy its
camera drivers (io/hikrobot, io/mindvision, io/usbcamera), so pictures look like what the robot sees.

The industrial cameras need their vendor SDK's Python wrapper, installed from the vendor's site:
    HikRobot:   MVS SDK -> /opt/MVS/Samples/<arch>/Python/MvImport   (or set MVS_PYTHON=/path/to/MvImport)
    MindVision: SDK     -> its demo/python folder with mvsdk.py      (or set MVSDK_PYTHON=/path/to/that/folder)
"""
import argparse
import ctypes
import glob
import os
import platform
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEMO_VIDEO = Path(os.environ.get("VIDEO", REPO / "videos" / "demo.avi")).expanduser()
OPENERS = {}


def register(prefix):
    """Make `prefix:arg` a valid --source. The opener gets (arg, **settings) and returns a Camera."""
    def wrap(fn):
        OPENERS[prefix] = fn
        return fn
    return wrap


class Camera:
    """Subclasses implement _grab(): return the next BGR frame, or None when there are no more."""
    kind = "camera"
    live = True  # False for files and folders: the same frames come back every time

    def __init__(self):
        self.info = {}

    def read(self):
        return self._grab()

    def frames(self, start=0, count=None, step=1):
        """Yield (index, frame), skipping the first `start` frames and keeping every `step`-th."""
        index, given = 0, 0
        while count is None or given < count:
            frame = self.read()
            if frame is None:
                return
            if index >= start and (index - start) % step == 0:
                yield index, frame
                given += 1
            index += 1

    def describe(self):
        return f"{self.kind}: " + ", ".join(f"{k} {v}" for k, v in self.info.items())

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ----------------------------------------------------------------------------- OpenCV-based sources
class OpenCvCamera(Camera):
    def __init__(self, cap, kind, live=True, what=""):
        super().__init__()
        if not cap.isOpened():
            raise RuntimeError(f"couldn't open {what or kind}")
        self.cap, self.kind, self.live = cap, kind, live
        self.info = {"size": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                     "fps": round(cap.get(cv2.CAP_PROP_FPS), 1), "backend": cap.getBackendName()}

    def _grab(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        self.cap.release()


def open_file(path, **_):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} doesn't exist")
    return OpenCvCamera(cv2.VideoCapture(str(path)), "video file", live=False, what=str(path))


def open_stream(url, **_):
    return OpenCvCamera(cv2.VideoCapture(url), "stream", what=url)


USB_BACKENDS = {"Linux": cv2.CAP_V4L2, "Windows": cv2.CAP_DSHOW, "Darwin": cv2.CAP_AVFOUNDATION}


@register("usb")
def open_usb(device="0", width=None, height=None, fps=None, exposure=None, gain=None, gamma=None, fourcc="MJPG"):
    """USB / UVC camera, set up like sp_vision's io/usbcamera/usbcamera.cpp."""
    backend = USB_BACKENDS.get(platform.system(), cv2.CAP_ANY)
    device = str(device)
    if device.startswith("video"):  # sp_vision configs say usb_device: "video0"
        device = "/dev/" + device
    cap = cv2.VideoCapture(int(device) if device.isdigit() else device, backend)
    if not cap.isOpened():
        raise RuntimeError(f"couldn't open USB camera {device}. See which ones exist: python3 shared/cameras.py --list")
    # Order matters: pixel format first (MJPG gets full resolution at full fps over USB), then size, then fps
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*str(fourcc)))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    if exposure is not None:
        # "manual exposure" is 1 for V4L2 (sp_vision sets 1) but 0.25 for DirectShow
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if backend == cv2.CAP_DSHOW else 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
    if gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, float(gain))
    if gamma is not None:
        cap.set(cv2.CAP_PROP_GAMMA, float(gamma))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # hand out the newest frame, not one that waited in a queue
    cam = OpenCvCamera(cap, "usb camera", what=f"USB camera {device}")
    asked = {k: v for k, v in dict(width=width, height=height, fps=fps).items() if v}
    if asked:
        cam.info["asked for"] = " ".join(f"{k}={v}" for k, v in asked.items())
    return cam


def gstreamer_available():
    return re.search(r"GStreamer:\s*YES", cv2.getBuildInformation()) is not None


@register("gst")
def open_gstreamer(pipeline, **_):
    if not gstreamer_available():
        raise RuntimeError("this OpenCV was built without GStreamer (pip's opencv-python never has it). "
                           "On the Jetson, use the system OpenCV that comes with JetPack.")
    return OpenCvCamera(cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER), "gstreamer", what=pipeline)


@register("csi")
def open_csi(sensor="0", width=1920, height=1080, fps=30, **_):
    """Jetson CSI camera: the ISP (nvarguscamerasrc) debayers on the GPU, nvvidconv copies to the CPU."""
    pipeline = (f"nvarguscamerasrc sensor-id={sensor} ! video/x-raw(memory:NVMM),width={width},height={height},"
                f"framerate={fps}/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=1 max-buffers=1")
    return open_gstreamer(pipeline)


class ImageFolder(Camera):
    kind, live = "images", False

    def __init__(self, files):
        super().__init__()
        if not files:
            raise FileNotFoundError("no images found")
        self.files, self.next = files, 0
        self.info = {"images": len(files)}

    def _grab(self):
        while self.next < len(self.files):
            img = cv2.imread(self.files[self.next])
            self.next += 1
            if img is not None:
                return img
        return None


# ----------------------------------------------------------------------------- industrial cameras (vendor SDKs)
def _import_sdk(module, env, folders, install_hint):
    for folder in [os.environ.get(env)] + folders:
        if folder and Path(folder).is_dir() and folder not in sys.path:
            sys.path.append(folder)
    try:
        return __import__(module)
    except ImportError as e:
        raise RuntimeError(f"{module}.py not found ({e}). {install_hint}, or set {env}=/folder/with/{module}.py") from None


class HikCamera(Camera):
    """HikRobot camera through the MVS SDK's Python wrapper, with the same settings as sp_vision's
    io/hikrobot/hikrobot.cpp: auto white balance on, auto exposure and gain off, Bayer -> BGR here."""
    kind = "hikrobot"

    def __init__(self, index="0", exposure_ms=2.0, gain=16.0, fps=None, **_):
        super().__init__()
        self.sdk = sdk = _import_sdk(
            "MvCameraControl_class", "MVS_PYTHON",
            ["/opt/MVS/Samples/aarch64/Python/MvImport", "/opt/MVS/Samples/64/Python/MvImport",
             r"C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport"],
            "Install HikRobot's MVS SDK (it brings the Python wrapper)")
        if hasattr(sdk.MvCamera, "MV_CC_Initialize"):  # MVS 4.x wants this before anything else
            sdk.MvCamera.MV_CC_Initialize()
        devices = sdk.MV_CC_DEVICE_INFO_LIST()
        self._check(sdk.MvCamera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, devices), "EnumDevices")
        if devices.nDeviceNum <= int(index):
            raise RuntimeError(f"found {devices.nDeviceNum} HikRobot camera(s), asked for number {index}")
        info = ctypes.cast(devices.pDeviceInfo[int(index)], ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
        self.cam = sdk.MvCamera()
        self._check(self.cam.MV_CC_CreateHandle(info), "CreateHandle")
        self._check(self.cam.MV_CC_OpenDevice(sdk.MV_ACCESS_Exclusive, 0), "OpenDevice")
        self.cam.MV_CC_SetEnumValue("TriggerMode", 0)       # free running
        self.cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 1)  # continuous
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)      # off
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)          # off
        self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_ms) * 1000)  # microseconds
        self.cam.MV_CC_SetFloatValue("Gain", float(gain))
        if fps:
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps))
        self._check(self.cam.MV_CC_StartGrabbing(), "StartGrabbing")
        self.buffer = sdk.MV_FRAME_OUT()
        # OpenCV names Bayer patterns one pixel off: these are the codes sp_vision uses, and they give BGR
        self.bayer = {getattr(sdk, f"PixelType_Gvsp_Bayer{p}8"): getattr(cv2, f"COLOR_Bayer{p}2RGB")
                      for p in ("GR", "RG", "GB", "BG") if hasattr(sdk, f"PixelType_Gvsp_Bayer{p}8")}
        self.info = {"exposure": f"{exposure_ms} ms", "gain": gain, "device": index}

    @staticmethod
    def _check(ret, what):
        if ret != 0:
            raise RuntimeError(f"HikRobot MV_CC_{what} failed: 0x{ret & 0xFFFFFFFF:08x}")

    def _grab(self):
        for _ in range(3):  # a few timeouts in a row means the camera is gone
            ctypes.memset(ctypes.byref(self.buffer), 0, ctypes.sizeof(self.buffer))
            if self.cam.MV_CC_GetImageBuffer(self.buffer, 1000) == 0:
                break
        else:
            return None
        info = self.buffer.stFrameInfo
        try:
            raw = np.ctypeslib.as_array(self.buffer.pBufAddr, shape=(info.nFrameLen,)).copy()
        finally:
            self.cam.MV_CC_FreeImageBuffer(self.buffer)
        h, w, kind = info.nHeight, info.nWidth, info.enPixelType
        if kind in self.bayer:
            return cv2.cvtColor(raw[:h * w].reshape(h, w), self.bayer[kind])
        if kind == getattr(self.sdk, "PixelType_Gvsp_Mono8", None):
            return cv2.cvtColor(raw[:h * w].reshape(h, w), cv2.COLOR_GRAY2BGR)
        if kind == getattr(self.sdk, "PixelType_Gvsp_BGR8_Packed", None):
            return raw[:h * w * 3].reshape(h, w, 3)
        if kind == getattr(self.sdk, "PixelType_Gvsp_RGB8_Packed", None):
            return cv2.cvtColor(raw[:h * w * 3].reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        raise RuntimeError(f"HikRobot pixel format 0x{kind:x} isn't handled: set the camera to an 8-bit Bayer "
                           "format in the MVS client, or add the conversion here")

    def close(self):
        self.cam.MV_CC_StopGrabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        if hasattr(self.sdk.MvCamera, "MV_CC_Finalize"):
            self.sdk.MvCamera.MV_CC_Finalize()


class MindVisionCamera(Camera):
    """MindVision camera through the SDK's mvsdk.py, with the same settings as sp_vision's
    io/mindvision/mindvision.cpp: auto exposure off, the SDK's ISP turns the raw image into BGR."""
    kind = "mindvision"

    def __init__(self, index="0", exposure_ms=2.0, gamma=0.5, **_):
        super().__init__()
        self.sdk = sdk = _import_sdk("mvsdk", "MVSDK_PYTHON", [], "Install the MindVision SDK (its demo/python has mvsdk.py)")
        devices = sdk.CameraEnumerateDevice()
        if len(devices) <= int(index):
            raise RuntimeError(f"found {len(devices)} MindVision camera(s), asked for number {index}")
        self.handle = sdk.CameraInit(devices[int(index)], -1, -1)
        cap = sdk.CameraGetCapability(self.handle)
        mono = cap.sIspCapacity.bMonoSensor != 0
        sdk.CameraSetIspOutFormat(self.handle, sdk.CAMERA_MEDIA_TYPE_MONO8 if mono else sdk.CAMERA_MEDIA_TYPE_BGR8)
        sdk.CameraSetTriggerMode(self.handle, 0)  # free running
        sdk.CameraSetAeState(self.handle, 0)      # auto exposure off
        sdk.CameraSetExposureTime(self.handle, float(exposure_ms) * 1000)  # microseconds
        sdk.CameraSetGamma(self.handle, int(float(gamma) * 100))
        sdk.CameraPlay(self.handle)
        size = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if mono else 3)
        self.frame_buffer = sdk.CameraAlignMalloc(size, 16)
        self.info = {"exposure": f"{exposure_ms} ms", "gamma": gamma, "device": index}

    def _grab(self):
        sdk = self.sdk
        for _ in range(3):
            try:
                raw, head = sdk.CameraGetImageBuffer(self.handle, 1000)
                break
            except sdk.CameraException:
                continue
        else:
            return None
        sdk.CameraImageProcess(self.handle, raw, self.frame_buffer, head)
        sdk.CameraReleaseImageBuffer(self.handle, raw)
        if platform.system() == "Windows":  # the Windows SDK delivers frames upside down
            sdk.CameraFlipFrameBuffer(self.frame_buffer, head, 1)
        data = (ctypes.c_ubyte * head.uBytes).from_address(self.frame_buffer)
        img = np.frombuffer(data, np.uint8).reshape(head.iHeight, head.iWidth, -1).copy()
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.shape[2] == 1 else img

    def close(self):
        self.sdk.CameraUnInit(self.handle)
        self.sdk.CameraAlignFree(self.frame_buffer)


@register("hik")
def open_hik(index="0", **settings):
    return HikCamera(index or "0", **settings)


@register("mv")
def open_mindvision(index="0", **settings):
    return MindVisionCamera(index or "0", **settings)


# ----------------------------------------------------------------------------- sp_vision configs
def read_config(path):
    """Top-level `key: value` lines of an sp_vision .yaml: numbers, "strings" and [lists] (no PyYAML needed)."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise SystemExit(f"config {path} doesn't exist")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z_]\w*):\s*(.*?)\s*(#.*)?$", line)
        if not m or not m.group(2):
            continue
        raw = m.group(2)
        if raw.startswith("["):
            values[m.group(1)] = [float(v) for v in raw.strip("[]").split(",") if v.strip()]
        elif raw[0] in "\"'":
            values[m.group(1)] = raw.strip("\"'")
        else:
            try:
                values[m.group(1)] = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                values[m.group(1)] = raw
    return values


def calibration_from_config(path):
    """(camera_matrix 3x3, dist_coeffs) from an sp_vision config, for PnP with that camera."""
    cfg = read_config(path)
    if "camera_matrix" not in cfg or "distort_coeffs" not in cfg:
        raise SystemExit(f"{path} has no camera_matrix / distort_coeffs")
    return np.array(cfg["camera_matrix"], np.float64).reshape(3, 3), np.array(cfg["distort_coeffs"], np.float64)


@register("config")
def open_config(path, **overrides):
    cfg = read_config(path)
    name = cfg.get("camera_name")
    if name == "usb":
        settings = dict(width=cfg.get("image_width"), height=cfg.get("image_height"), fps=cfg.get("usb_frame_rate"),
                        exposure=cfg.get("usb_exposure"), gain=cfg.get("usb_gain"), gamma=cfg.get("usb_gamma"))
        return open_usb(str(cfg.get("usb_device", "0")), **{**settings, **overrides})
    if name == "hikrobot":
        return HikCamera(**{"exposure_ms": cfg.get("exposure_ms", 2.0), "gain": cfg.get("gain", 16.0), **overrides})
    if name == "mindvision":
        return MindVisionCamera(**{"exposure_ms": cfg.get("exposure_ms", 2.0), "gamma": cfg.get("gamma", 0.5), **overrides})
    raise SystemExit(f"{path}: camera_name {name!r} isn't usb, hikrobot or mindvision")


# ----------------------------------------------------------------------------- picking a source
def open_camera(source=None, **settings):
    """Open whatever `source` describes (see the top of this file). None means the demo video."""
    source = str(source if source is not None else DEMO_VIDEO)
    if "://" in source:
        return open_stream(source, **settings)
    if source in OPENERS:  # just "hik", "usb", ...
        return OPENERS[source](**settings)
    prefix, sep, arg = source.partition(":")
    if sep and prefix in OPENERS and not re.match(r"^[A-Za-z]$", prefix):  # "C:\..." is a Windows path, not a prefix
        return OPENERS[prefix](arg, **settings) if arg else OPENERS[prefix](**settings)
    if source.isdigit() or source.startswith("/dev/video"):
        return open_usb(source, **settings)
    if Path(source).is_dir():
        files = [f for ext in ("jpg", "jpeg", "png", "bmp") for f in glob.glob(str(Path(source) / f"*.{ext}"))]
        return ImageFolder(sorted(files))
    if any(c in source for c in "*?["):
        return ImageFolder(sorted(glob.glob(source)))
    return open_file(source, **settings)


def _value(text):
    try:
        return float(text) if any(c in text for c in ".eE") else int(text)
    except ValueError:
        return text


def add_source_args(ap):
    """--source and --cam for any script. Read them back with source_from_args()."""
    ap.add_argument("--source", help="camera, video, folder or stream (see cameras.py). Default: the demo video")
    ap.add_argument("--video", help=argparse.SUPPRESS)  # the old name of --source
    ap.add_argument("--cam", action="append", default=[], metavar="KEY=VALUE",
                    help="camera setting, repeatable: width=1280, fps=60, exposure_ms=2, ...")


def source_from_args(args):
    settings = {}
    for item in args.cam:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--cam wants KEY=VALUE, got {item!r}")
        settings[key.strip()] = _value(value.strip())
    return args.source or args.video, settings


# ----------------------------------------------------------------------------- preview / record / list
def list_cameras(max_usb=8):
    backend = USB_BACKENDS.get(platform.system(), cv2.CAP_ANY)
    print(f"OpenCV {cv2.__version__}, GStreamer {'yes' if gstreamer_available() else 'no'} (csi: and gst: need it)")
    found = 0
    for i in range(max_usb):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            size = f"{frame.shape[1]}x{frame.shape[0]}" if ok else "opens, but no frame"
            print(f"  usb:{i}   {size} by default, backend {cap.getBackendName()}")
            found += 1
        cap.release()
    if not found:
        print("  no USB cameras")
    for name, cls in (("hik", HikCamera), ("mv", MindVisionCamera)):
        try:
            cam = cls()
            print(f"  {name}:0   {cam.describe()}")
            cam.close()
        except Exception as e:  # SDK missing, library won't load, no camera plugged in...
            print(f"  {name}: {str(e).splitlines()[0]}")


def preview(cam, record=None, seconds=None, headless=False):
    """Show (and/or record) frames. Keys: q quit, s snapshot, r start/stop recording."""
    out_dir = Path("out")  # next to wherever you run it from
    out_dir.mkdir(exist_ok=True)
    writer, path, shown, t0, fps = None, None, 0, time.perf_counter(), 0.0

    def start(p):
        frame_fps = cam.info.get("fps") or 30
        return cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"MJPG"), frame_fps if frame_fps > 0 else 30, size), p

    size = None
    last = time.perf_counter()
    while seconds is None or time.perf_counter() - t0 < seconds:
        frame = cam.read()
        if frame is None:
            print("no more frames")
            break
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now
        size = (frame.shape[1], frame.shape[0])
        if record and writer is None:
            writer, path = start(Path(record))
            print(f"recording to {path}")
        if writer:
            writer.write(frame)
        shown += 1
        if headless:
            if shown % 30 == 0:
                print(f"{shown} frames, {fps:.0f} FPS", flush=True)
            continue
        view = frame.copy()
        status = f"{size[0]}x{size[1]}  {fps:.0f} FPS" + ("  REC" if writer else "") + "   q quit  s snapshot  r record"
        cv2.putText(view, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if writer else (0, 255, 0), 2)
        cv2.imshow("camera", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            snap = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(snap), frame)
            print(f"saved {snap}")
        if key == ord("r"):
            if writer:
                writer.release()
                print(f"saved {path}")
                writer = None
            else:
                writer, path = start(out_dir / f"record_{time.strftime('%Y%m%d_%H%M%S')}.avi")
                print(f"recording to {path}")
    if writer:
        writer.release()
        print(f"saved {path}")
    print(f"{shown} frames in {time.perf_counter() - t0:.1f} s")
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Preview, record or list cameras.")
    add_source_args(ap)
    ap.add_argument("--list", action="store_true", help="list the cameras this machine can open")
    ap.add_argument("--record", help="record to this .avi from the start")
    ap.add_argument("--seconds", type=float, help="stop after this long")
    ap.add_argument("--headless", action="store_true", help="no window (over SSH): use with --record")
    args = ap.parse_args()
    if args.list:
        list_cameras()
        return
    source, settings = source_from_args(args)
    try:
        cam = open_camera(source, **settings)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"Couldn't open {source or DEMO_VIDEO}: {e}") from None
    with cam:
        print(cam.describe())
        preview(cam, args.record, args.seconds, args.headless)


if __name__ == "__main__":
    main()
