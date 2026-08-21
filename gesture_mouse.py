# -*- coding: utf-8 -*-
"""
手势鼠标 GestureMouse —— 用摄像头识别手势控制鼠标
==================================================
双引擎手部识别：
  1. MediaPipe（高精度 21 个关键点，需 pip install mediapipe）
  2. OpenCV 肤色分割 + 凸包缺陷法（零额外依赖，作为回退）

鼠标控制：Win32 API（ctypes 直调 SetCursorPos / mouse_event），不依赖 pyautogui。

手势定义：
  1 根手指（食指）                -> 移动光标
  2 根手指（食指+中指）           -> 左键单击
  3 根手指（食中无名）            -> 右键单击
  4 根手指（无拇指）              -> 滚轮（手高于画面中线向上滚，低于则向下）
  5 根手指（张开手掌）            -> 按住左键拖拽（合拢后松开）
  拇指+食指捏合（MediaPipe 引擎）-> 按住左键拖拽（松开捏合即释放）
  手离开画面 / 无法识别           -> 鼠标静止，不误触

命令行：
  python gesture_mouse.py            正常启动图形界面
  python gesture_mouse.py --smoke    图形界面冒烟测试（3 秒后自动关闭）
  python gesture_mouse.py --headless-test  无界面自检（真实摄像头跑数秒并输出诊断）
"""

import sys
import os
import traceback

# 崩溃日志：模块级尽早安装（打包版 windowed 无控制台，异常内容只能落盘）
_CRASH_BASE = (os.path.dirname(os.path.abspath(sys.executable))
               if getattr(sys, "frozen", False)
               else os.path.dirname(os.path.abspath(__file__)))


def _excepthook(tp, value, tb):
    try:
        with open(os.path.join(_CRASH_BASE, "crash_log.txt"),
                  "w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(tp, value, tb)))
    except Exception:
        pass
    sys.__excepthook__(tp, value, tb)


sys.excepthook = _excepthook

import ctypes
import ctypes.wintypes   # 注意：只 import ctypes 不会加载 wintypes 子模块
import json
import math
import queue
import threading
import time

import cv2
import numpy as np

APP_NAME = "手势鼠标 GestureMouse"
VERSION = "2.0.0"

if getattr(sys, "frozen", False):
    # PyInstaller 打包运行：资源（模型/设置）与 exe 同目录
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")

DEFAULT_SETTINGS = {
    "camera": "auto",          # 摄像头索引：auto / 0 / 1 / 2 / 3
    "engine": "auto",          # auto / mediapipe / skin
    "hands_mode": "dual",      # dual 双手模式 / single 单手模式
    "head_enabled": False,     # 头动光标独立开关：True 时头部朝向控制光标（与手势解耦）
    "sensitivity": 1.0,        # 光标灵敏度（绝对模式：映射倍数；相对模式：速度倍率）
    "touch_sensitivity": 0.55, # 触摸灵敏度：拇指与指尖判定为"触摸"的距离阈值倍数
    "scroll_speed": 1.0,      # 滚轮速度倍率（4 指滚轮 / 拇指触摸上下滑）
    "head_sensitivity": 1.0,   # 头动模式：头部转动放大增益（角度域，未标定时生效）
    "head_calib": None,        # 头动标定系数（12 个浮点，标定后自动写入）
    "head_pose_v2": True,      # 姿态估计版本标记（矩阵角度版）
    "head_recenter": False,    # 运行时标志：重置头动零点（GUI 按钮触发）
    "game_mode": False,        # 游戏模式（战争雷霆等）：SendInput 输入+游戏手势映射
    "smoothing": 0.35,         # 平滑度 EMA 系数
    "mode": "absolute",        # absolute 绝对定位 / relative 相对移动（触摸板式）
    "mirror": True,            # 镜像画面
    "show_landmarks": True,    # 绘制手部标记
    "show_hint": True,         # 画面角标提示
    "monitor_enabled": True,   # 屏幕右下角悬浮监视窗
    "monitor_mode": "full",    # full 都看 / raw 只看动画 / marker 只看标记
    "monitor_w": 320,          # 监视窗初始宽度
    "monitor_h": 240,          # 监视窗初始高度
}

GESTURE_NONE = "none"
GESTURE_MOVE = "move"
GESTURE_LCLICK = "lclick"
GESTURE_RCLICK = "rclick"
GESTURE_DCLICK = "dclick"
GESTURE_SCROLL = "scroll"
GESTURE_SCROLLUP = "scrollup"
GESTURE_SCROLLDOWN = "scrolldown"
GESTURE_DRAG = "drag"
GESTURE_KEY_TAB = "keytab"   # 游戏模式：5 指张开 = Tab（锁定目标）

GESTURE_LABEL_CN = {
    GESTURE_NONE: "无动作",
    GESTURE_MOVE: "移动光标",
    GESTURE_LCLICK: "左键单击",
    GESTURE_RCLICK: "右键单击",
    GESTURE_DCLICK: "左键双击",
    GESTURE_SCROLL: "滚轮",
    GESTURE_SCROLLUP: "上滑（向上滚动）",
    GESTURE_SCROLLDOWN: "下滑（向下滚动）",
    GESTURE_DRAG: "按住拖拽",
    GESTURE_KEY_TAB: "Tab（锁定目标）",
}

# 监视窗动作说明（手势名 -> 具体怎么做）
GESTURE_HINT_CN = {
    GESTURE_MOVE: "右手食指尖移动光标",
    GESTURE_LCLICK: "拇+食指短触摸",
    GESTURE_RCLICK: "拇+中指触摸",
    GESTURE_DCLICK: "快速两连捏",
    GESTURE_SCROLL: "4 指滚轮（单手模式）",
    GESTURE_SCROLLUP: "拇+无名指触摸",
    GESTURE_SCROLLDOWN: "拇+小指触摸",
    GESTURE_DRAG: "拇+食指长按拖拽",
    GESTURE_NONE: "握拳 / 5 指张开 / 无手",
}

# 监视窗动作颜色
GESTURE_COLOR_CN = {
    GESTURE_MOVE: "#4caf50",
    GESTURE_LCLICK: "#ffd54f",
    GESTURE_RCLICK: "#ff9800",
    GESTURE_DCLICK: "#ffeb3b",
    GESTURE_SCROLL: "#42a5f5",
    GESTURE_SCROLLUP: "#29b6f6",
    GESTURE_SCROLLDOWN: "#26c6da",
    GESTURE_DRAG: "#ef5350",
    GESTURE_NONE: "#9e9e9e",
}


# ---------------------------------------------------------------------------
# Win32 输入控制（SendInput，游戏可识别）
# ---------------------------------------------------------------------------
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("union", _INPUTUNION)]


class MouseController:
    """用 ctypes 直调 Win32 API 控制鼠标与键盘。

    输入注入统一走 SendInput（老 mouse_event 会被部分游戏忽略）：
    - 绝对移动：SetCursorPos（战争雷霆鼠标瞄准等跟随系统光标的游戏）
    - 相对移动：SendInput 增量运动（DirectInput/原生输入的游戏必须用这个，
      移动系统光标它们不认）
    - 点击/滚轮/按键：SendInput
    """

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120
    KEYEVENTF_KEYUP = 0x0002
    VK_TAB = 0x09

    def __init__(self):
        self.user32 = ctypes.windll.user32
        self.screen_w = int(self.user32.GetSystemMetrics(0))
        self.screen_h = int(self.user32.GetSystemMetrics(1))
        try:
            self.dclick_ms = int(self.user32.GetDoubleClickTime())
        except Exception:
            self.dclick_ms = 500
        self.dclick_ms = max(200, min(900, self.dclick_ms))
        self._extra = ctypes.c_ulong(0)

    # -- SendInput 底层 -----------------------------------------------------
    def _send_mouse(self, flags, dx=0, dy=0, wheel=0):
        inp = _INPUT()
        inp.type = 0  # INPUT_MOUSE
        inp.union.mi = _MOUSEINPUT(int(dx), int(dy), int(wheel), int(flags),
                                   0, ctypes.pointer(self._extra))
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def _send_key(self, vk, up=False):
        inp = _INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.union.ki = _KEYBDINPUT(int(vk), 0,
                                   self.KEYEVENTF_KEYUP if up else 0,
                                   0, ctypes.pointer(self._extra))
        self.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))

    def position(self):
        pt = ctypes.wintypes.POINT()
        self.user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def move_abs(self, nx, ny):
        """nx, ny 为 0~1 归一化坐标（相对主屏）。"""
        x = int(round(min(1.0, max(0.0, nx)) * (self.screen_w - 1)))
        y = int(round(min(1.0, max(0.0, ny)) * (self.screen_h - 1)))
        self.user32.SetCursorPos(x, y)

    def move_rel(self, dx, dy):
        """相对移动：SendInput 增量运动（游戏可识别）。"""
        self._send_mouse(self.MOUSEEVENTF_MOVE, int(round(dx)), int(round(dy)))

    def click_left(self):
        self._send_mouse(self.MOUSEEVENTF_LEFTDOWN)
        time.sleep(0.012)
        self._send_mouse(self.MOUSEEVENTF_LEFTUP)

    def click_right(self):
        self._send_mouse(self.MOUSEEVENTF_RIGHTDOWN)
        time.sleep(0.012)
        self._send_mouse(self.MOUSEEVENTF_RIGHTUP)

    def double_click(self):
        """双击：两次点击间隔自适应系统双击速度设置。

        间隔取系统双击时间的 1/4（clamp 40~150ms），既保证系统能
        识别为双击，又不会慢得让用户觉得拖沓。
        """
        interval = min(0.15, max(0.04, self.dclick_ms * 0.00025))
        self.click_left()
        time.sleep(interval)
        self.click_left()

    def left_down(self):
        self._send_mouse(self.MOUSEEVENTF_LEFTDOWN)

    def left_up(self):
        self._send_mouse(self.MOUSEEVENTF_LEFTUP)

    def scroll(self, clicks):
        """clicks 正=向上滚，负=向下滚。"""
        self._send_mouse(self.MOUSEEVENTF_WHEEL,
                         wheel=int(clicks * self.WHEEL_DELTA))

    def key_tap(self, vk_code):
        """按一下键盘键（游戏模式用，如 Tab 锁定目标）。SendInput 注入。"""
        self._send_key(vk_code)
        time.sleep(0.03)
        self._send_key(vk_code, up=True)


class DryRunMouse(MouseController):
    """测试用：不碰真实鼠标，只记录动作。"""

    def __init__(self):
        super().__init__()
        self.events = {"move": 0, "lclick": 0, "rclick": 0, "dclick": 0,
                       "scroll": 0, "down": 0, "up": 0, "scroll_sum": 0,
                       "key": 0}
        self.last_pos = (0.5, 0.5)

    def key_tap(self, vk_code):
        self.events["key"] += 1

    def move_abs(self, nx, ny):
        self.events["move"] += 1
        self.last_pos = (nx, ny)

    def move_rel(self, dx, dy):
        self.events["move"] += 1

    def click_left(self):
        self.events["lclick"] += 1

    def click_right(self):
        self.events["rclick"] += 1

    def double_click(self):
        self.events["dclick"] += 1

    def left_down(self):
        self.events["down"] += 1

    def left_up(self):
        self.events["up"] += 1

    def scroll(self, clicks):
        self.events["scroll"] += 1
        self.events["scroll_sum"] += int(clicks)


# ---------------------------------------------------------------------------
# 识别结果结构
# ---------------------------------------------------------------------------
class HandInfo:
    """单只手的信息。label 为用户视角：'left' / 'right'。"""

    def __init__(self):
        self.found = False
        self.label = ""                 # 用户视角左右手
        self.fingers = [False] * 5      # [拇指, 食指, 中指, 无名指, 小指]
        self.pointer = (0.5, 0.5)       # 归一化光标目标点（通常为食指尖）
        self.pinch = False              # 兼容字段：拇指+食指触摸
        self.thumb_touch = 0            # 0=无触摸 1=食指 2=中指 3=无名指 4=小指
        self.landmarks = None           # MediaPipe 关键点 [(x, y), ...] 归一化

    @property
    def count(self):
        return sum(self.fingers)


class GestureResult:
    def __init__(self):
        self.hands = []                 # HandInfo 列表（0~2 只手）
        self.engine_name = ""
        # 头动模式数据
        self.gaze_point = None          # (rx, ry) 头部朝向比例（0~1）
        self.gaze_valid = False         # 当前帧姿态有效（有人脸）
        self.face_found = False
        self.blink = False              # 兼容字段（头动模式恒 False）
        self.face_pts = []              # [左眼角, 右眼角, 眼宽中点, 鼻尖] 归一化点
        self.face_landmarks = None      # 全部 468 个脸部关键点（画面绘制用）

    def hand(self, label):
        """取用户视角的某只手；label: 'left' / 'right'。"""
        for h in self.hands:
            if h.label == label:
                return h
        return None

    def main_hand(self):
        """主手：有手时返回第一只手，否则返回一个空 HandInfo。"""
        return self.hands[0] if self.hands else HandInfo()

    # ---- 兼容旧单手接口（指向主手）----
    @property
    def hand_found(self):
        return len(self.hands) > 0

    @property
    def fingers(self):
        return self.main_hand().fingers

    @property
    def pointer(self):
        return self.main_hand().pointer

    @property
    def pinch(self):
        return self.main_hand().pinch

    @property
    def landmarks(self):
        return self.main_hand().landmarks

    @property
    def count(self):
        return self.main_hand().count


def _to_short_path(path):
    """中文路径转 Windows 8.3 短路径（mediapipe C API 不认中文路径）。"""
    try:
        if sys.platform == "win32" and not path.isascii():
            buf = ctypes.create_unicode_buffer(512)
            n = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
            if 0 < n < 512:
                return buf.value
    except Exception:
        pass
    return path


# ---------------------------------------------------------------------------
# 引擎 1：MediaPipe
# ---------------------------------------------------------------------------
class MediaPipeEngine:
    name = "MediaPipe"
    detail = "高精度 21 关键点（推荐，需安装 mediapipe 包）"
    MODEL_NAME = "hand_landmarker.task"

    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError:
            raise RuntimeError(
                "MediaPipe 未安装。可执行: pip install mediapipe\n"
                "若 Python 版本过新没有预编译包，请改用肤色识别引擎。"
            )
        self.mp = mp
        self.vision = vision
        model_path = os.path.join(BASE_DIR, self.MODEL_NAME)
        if not os.path.isfile(model_path):
            raise RuntimeError(
                "缺少模型文件 %s，请将其放到程序目录下。\n"
                "下载地址：https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                % self.MODEL_NAME
            )
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_to_short_path(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts = 0

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    @staticmethod
    def _fingers_up(lm):
        """手指伸展判定（旋转不变、外展容忍、握拳不误判）：

        - 拇指：拇指尖到小指掌根的距离明显大于到食指掌根的距离（>1.15 倍）
        - 其余四指同时满足：
          1) 距离判据：指尖到手腕的距离明显大于其关节到手腕的距离（>1.08 倍）
          2) 方向判据：指尖-关节向量与手方向夹角 < 135°（容忍外展，
             但握拳时指尖卷向手心约 180°，判为未伸出）
        """
        up = [False] * 5
        d17 = math.hypot(lm[4].x - lm[17].x, lm[4].y - lm[17].y)
        d5 = math.hypot(lm[4].x - lm[5].x, lm[4].y - lm[5].y)
        up[0] = d17 > 1.15 * max(d5, 1e-6)
        ux = lm[9].x - lm[0].x
        uy = lm[9].y - lm[0].y
        for i, (tip, pip) in enumerate(((8, 6), (12, 10), (16, 14), (20, 18)),
                                       start=1):
            d_tip = math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y)
            d_pip = math.hypot(lm[pip].x - lm[0].x, lm[pip].y - lm[0].y)
            dist_ok = d_tip > 1.08 * max(d_pip, 1e-6)
            vx = lm[tip].x - lm[pip].x
            vy = lm[tip].y - lm[pip].y
            dot = vx * ux + vy * uy
            mag = math.hypot(vx, vy) * math.hypot(ux, uy)
            angle_ok = dot > -0.7 * max(mag, 1e-9)   # 夹角 < 135°
            up[i] = dist_ok and angle_ok
        return up

    @staticmethod
    def _thumb_touch(lm, threshold=0.55, z_weight=1.0):
        """拇指触摸判定：用 3D 距离（含深度 z）。

        手旋转时拇指与指尖在画面上的 2D 投影会重叠，但真实深度不同；
        加入 z 维度后旋转不再误判为触摸。
        返回 0=无触摸 1=食指 2=中指 3=无名指 4=小指。
        """
        ref = math.sqrt((lm[0].x - lm[9].x) ** 2
                        + (lm[0].y - lm[9].y) ** 2
                        + ((lm[0].z - lm[9].z) * z_weight) ** 2)
        ref = max(ref, 1e-6)
        dists = []
        for i, tip_id in enumerate((8, 12, 16, 20), start=1):
            dx = lm[4].x - lm[tip_id].x
            dy = lm[4].y - lm[tip_id].y
            dz = (lm[4].z - lm[tip_id].z) * z_weight
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            dists.append((d, i))
        dists.sort(key=lambda t: t[0])
        best_d, best_i = dists[0]
        second_d = dists[1][0]
        # 显著近于第二近的指尖（防止做"拇+小指"时弯曲的食指抢走判定）
        if best_d < threshold * ref and best_d < 0.82 * second_d:
            return best_i
        return 0

    def detect(self, frame_rgb, mirror=True, touch_threshold=0.55):
        """检测最多两只手。

        mirror=True 表示输入图像已镜像（摄像头预览镜像），此时
        MediaPipe 的左右手标注与实际相反，需要翻转成用户视角。
        touch_threshold: 触摸判定距离阈值倍数（越小越需贴紧，越大越灵敏）。
        """
        res = GestureResult()
        res.engine_name = self.name

        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(frame_rgb))
        ts = int(time.monotonic() * 1000)
        if ts <= self._ts:      # VIDEO 模式要求时间戳单调递增
            ts = self._ts + 1
        self._ts = ts
        result = self.landmarker.detect_for_video(img, ts)

        if not result.hand_landmarks:
            return res

        for lm, hc in zip(result.hand_landmarks, result.handedness):
            hi = HandInfo()
            hi.found = True
            media_label = hc[0].category_name          # "Left" / "Right"
            if mirror:
                hi.label = "right" if media_label == "Left" else "left"
            else:
                hi.label = media_label.lower()

            # 关键点含深度 z（用于旋转鲁棒的触摸判定）
            hi.landmarks = [(p.x, p.y, p.z) for p in lm]

            # 手指伸展判定（手方向投影，旋转不变）
            hi.fingers = self._fingers_up(lm)

            # 光标点：食指尖（食指未伸出时退化为指尖与掌根中点的中点）
            if hi.fingers[1]:
                hi.pointer = (lm[8].x, lm[8].y)
            else:
                hi.pointer = (
                    0.5 * (lm[8].x + lm[9].x),
                    0.5 * (lm[8].y + lm[9].y),
                )

            # 拇指触摸判定（3D 距离，旋转不误判）
            hi.thumb_touch = self._thumb_touch(lm, touch_threshold)
            hi.pinch = (hi.thumb_touch == 1)
            res.hands.append(hi)
        return res

    def annotate(self, frame_bgr, res):
        """在 BGR 帧上叠加标记，返回新帧。"""
        return frame_bgr


# ---------------------------------------------------------------------------
# 引擎 2：OpenCV 肤色分割 + 凸包缺陷数手指（回退引擎）
# ---------------------------------------------------------------------------
class SkinEngine:
    name = "肤色识别"
    detail = "OpenCV 肤色分割 + 凸包缺陷法（无需 mediapipe，对光线有一定要求）"

    SKIN_MIN = np.array([0, 133, 77], dtype=np.uint8)    # YCrCb
    SKIN_MAX = np.array([255, 173, 127], dtype=np.uint8)

    def __init__(self):
        pass

    def close(self):
        pass

    def detect(self, frame_bgr, mirror=True):
        """检测最多两只手（取两个最大肤色轮廓，按画面 x 位置判定左右）。

        mirror=True（画面镜像）：x 小 = 用户左手；否则 x 小 = 用户右手。
        """
        res = GestureResult()
        res.engine_name = self.name
        h, w = frame_bgr.shape[:2]

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.SKIN_MIN, self.SKIN_MAX)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return res

        # 按面积取前两个候选，按 x 排序
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        contours.sort(key=lambda c: cv2.boundingRect(c)[0])
        for i, cnt in enumerate(contours):
            if mirror:
                label = "left" if i == 0 else "right"
            else:
                label = "right" if i == 0 else "left"
            hi = self._analyze_contour(cnt, h, w, label)
            if hi is not None:
                res.hands.append(hi)
        return res

    def _analyze_contour(self, cnt, h, w, label):
        area = cv2.contourArea(cnt)
        frame_area = float(h * w)
        if area < 0.015 * frame_area or area > 0.7 * frame_area:
            return None

        # 手形长宽比粗筛（去掉脸/胳膊等大块）
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bh / max(bw, 1)
        if aspect < 0.5 or aspect > 2.8:
            return None

        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            return None
        cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

        hull = cv2.convexHull(cnt)
        hull_idx = cv2.convexHull(cnt, returnPoints=False)

        # 指尖候选：凸包上尖锐顶点、远离掌心、位于掌心上部
        tips = []
        max_d = 0.0
        for p in hull:
            d = math.hypot(p[0][0] - cx, p[0][1] - cy)
            max_d = max(max_d, d)
        for i in range(len(hull)):
            a = hull[i - 1][0]
            p = hull[i][0]
            b = hull[(i + 1) % len(hull)][0]
            ang = _angle_between(a, p, b)
            d = math.hypot(p[0] - cx, p[1] - cy)
            if ang < 95 and d > 0.5 * max_d and p[1] < cy + 0.15 * max_d:
                tips.append(p)

        # 凸包缺陷：指缝数量 => 伸出指数 = 缺陷数 + 1
        n_defects = 0
        far_points = []
        if len(hull_idx) >= 5:
            defects = cv2.convexityDefects(cnt, hull_idx)
            if defects is not None:
                for i in range(defects.shape[0]):
                    # OpenCV 4: shape (N,1,4)；OpenCV 5: shape (N,4)
                    row = defects[i]
                    if getattr(row, "ndim", 1) > 1:
                        row = row[0]
                    try:
                        s, e, f, depth = (int(row[0]), int(row[1]),
                                          int(row[2]), int(row[3]))
                    except Exception:
                        continue
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])
                    a_ = math.hypot(start[0] - far[0], start[1] - far[1])
                    b_ = math.hypot(end[0] - far[0], end[1] - far[1])
                    c_ = math.hypot(start[0] - end[0], start[1] - end[1])
                    if c_ <= 0:
                        continue
                    cos_v = (a_ * a_ + b_ * b_ - c_ * c_) / (2 * a_ * b_)
                    cos_v = max(-1.0, min(1.0, cos_v))
                    ang = math.degrees(math.acos(cos_v))
                    if ang < 90 and depth > 0.16 * max_d:
                        n_defects += 1
                        far_points.append(far)

        count = 0
        if tips:
            count = min(5, max(1, n_defects + 1))
            if count == 1 and n_defects == 0 and len(tips) == 1:
                pass  # 单指尖且无指缝：视为 1 指

        if count == 0:
            return None

        hi = HandInfo()
        hi.found = True
        hi.label = label
        # 约定按 [拇指,食指,中指,无名指,小指] 累计标记
        hi.fingers = [count >= 5, count >= 1, count >= 2, count >= 3, count >= 4]

        # 光标点：取最上方的指尖（该引擎的近似方案，MediaPipe 下会更精确）
        tip_top = min(tips, key=lambda p: p[1])
        hi.pointer = (tip_top[0] / w, tip_top[1] / h)
        hi.landmarks = None
        return hi

    def annotate(self, frame_bgr, res):
        return frame_bgr


def _angle_between(a, p, b):
    v1 = (a[0] - p[0], a[1] - p[1])
    v2 = (b[0] - p[0], b[1] - p[1])
    d1 = math.hypot(*v1)
    d2 = math.hypot(*v2)
    if d1 == 0 or d2 == 0:
        return 180.0
    cos_v = (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)
    cos_v = max(-1.0, min(1.0, cos_v))
    return math.degrees(math.acos(cos_v))


# ---------------------------------------------------------------------------
# 引擎 3：头动追踪（MediaPipe FaceLandmarker 关键点姿态）
# ---------------------------------------------------------------------------
class PoseCalibrator:
    """头动标定：二次多项式岭回归拟合 (rx, ry) -> 屏幕坐标。

    每个人的坐姿、摄像头位置、头部活动习惯不同，固定增益必然不准；
    标定后按个人头部朝向特征映射，指向精度大幅提高。
    """

    @staticmethod
    def fit(points, alpha=1e-4):
        """points: [(rx, ry, sx, sy), ...] 至少 6 个 -> 12 个系数或 None。"""
        if len(points) < 6:
            return None
        A = np.zeros((len(points), 6), dtype=np.float64)
        for i, (rx, ry, sx, sy) in enumerate(points):
            A[i] = [1.0, rx, ry, rx * ry, rx * rx, ry * ry]
        bx = np.array([p[2] for p in points], dtype=np.float64)
        by = np.array([p[3] for p in points], dtype=np.float64)
        try:
            reg = A.T @ A + alpha * np.eye(6)
            cx = np.linalg.solve(reg, A.T @ bx)
            cy = np.linalg.solve(reg, A.T @ by)
        except Exception:
            return None
        return [float(v) for v in list(cx) + list(cy)]

    @staticmethod
    def predict(coeffs, rx, ry):
        """coeffs: 12 个系数（fit 输出）-> (sx, sy) 屏幕归一化坐标。"""
        v = [1.0, rx, ry, rx * ry, rx * rx, ry * ry]
        sx = sum(c * t for c, t in zip(coeffs[0:6], v))
        sy = sum(c * t for c, t in zip(coeffs[6:12], v))
        return min(1.0, max(0.0, sx)), min(1.0, max(0.0, sy))


class HeadPoseEngine:
    """头动姿态：MediaPipe 官方人脸变换矩阵 -> 欧拉角（弧度）。

    相比鼻尖-眼角三点几何法，变换矩阵是完整头部姿态估计，
    对头部侧转时的透视变化有补偿，指向更准、方向更稳。
    返回 (yaw, pitch) 弧度：0 = 正对摄像头，正 = 右转/抬头。
    """

    name = "头动追踪"
    MODEL_NAME = "face_landmarker.task"

    # 关键点（画面标注用）：鼻尖(1)、左外眼角(33)、右外眼角(263)
    NOSE = 1
    EYE_L, EYE_R = 33, 263

    def __init__(self):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError:
            raise RuntimeError("头动模式需要 mediapipe 包（未安装）")
        self.mp = mp
        self.vision = vision
        model_path = os.path.join(BASE_DIR, self.MODEL_NAME)
        if not os.path.isfile(model_path):
            raise RuntimeError(
                "缺少模型文件 %s，请将其放到程序目录下。\n"
                "下载地址：https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                % self.MODEL_NAME
            )
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=_to_short_path(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        self._ts = 0

    def close(self):
        try:
            self.landmarker.close()
        except Exception:
            pass

    @staticmethod
    def _angles_from_matrix(M):
        """4x4 变换矩阵 -> (yaw, pitch) 弧度。

        旋转轴对应关系（用户正对摄像头）：
        - 左右转头 = 绕竖直轴（相机 y 轴）
        - 上下点头 = 绕水平轴（相机 x 轴）——即 R[2,1]/R[2,2]
        头部侧倾（绕视线轴）忽略。
        """
        R = np.asarray(M, dtype=np.float64)[:3, :3]
        yaw = math.atan2(R[1, 0], R[0, 0])     # 绕竖直轴：左右转头
        pitch = math.atan2(R[2, 1], R[2, 2])   # 绕水平轴：上下点头
        return float(yaw), float(pitch)

    def detect(self, frame_rgb, mirror=True):
        """返回 (pose, info)。

        pose: (yaw, pitch) 弧度；0 = 正对摄像头。
              yaw 正 = 右转，pitch 正 = 抬头。
        mirror=True（画面镜像）时 yaw 符号翻转，保持"用户视角方向"一致。
        info: {"face": bool, "face_pts": [...], "landmarks": [...]}
        """
        info = {"face": False, "face_pts": [], "landmarks": None}
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=np.ascontiguousarray(frame_rgb))
        ts = int(time.monotonic() * 1000)
        if ts <= self._ts:
            ts = self._ts + 1
        self._ts = ts
        result = self.landmarker.detect_for_video(img, ts)
        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return None, info
        info["face"] = True

        yaw, pitch = self._angles_from_matrix(
            result.facial_transformation_matrixes[0])
        if mirror:
            yaw = -yaw

        # 关键点供画面标注（归一化坐标）：左眼角, 右眼角, 眼宽中点, 鼻尖
        lm = result.face_landmarks[0]
        lx, ly = lm[self.EYE_L].x, lm[self.EYE_L].y
        rx_, ry_ = lm[self.EYE_R].x, lm[self.EYE_R].y
        info["face_pts"] = ((lx, ly), (rx_, ry_),
                            ((lx + rx_) / 2.0, (ly + ry_) / 2.0),
                            (lm[self.NOSE].x, lm[self.NOSE].y))
        # 全部 468 个关键点（供画面绘制网格与特征点）
        info["landmarks"] = [(p.x, p.y) for p in lm]
        return (float(yaw), float(pitch)), info


class HeadController:
    """独立头动光标控制器（与手势解释器完全解耦）。

    - 岭回归个人映射（标定后），未标定时线性+增益回退
    - 多帧中值聚合（抑制姿态检测抖动）+ EMA 平滑 + 死区
    - 无人脸时冻结光标，绝不误动
    本类只负责"头指向哪光标去哪"，不感知任何手势逻辑。
    """

    MEDIAN_WIN = 3          # 中值聚合窗口（帧）——3 帧滞后约 33ms，5 帧太拖
    DEAD_ZONE = 0.002       # 目标偏差死区（屏幕归一化坐标，约 4px@1920）

    def __init__(self, settings, mouse):
        self.settings = settings
        self.mouse = mouse
        self.engine = None
        self.px, self.py = 0.5, 0.5
        self._recent = []
        # 零点偏移：姿态角零点是"脸正对摄像头"，自然坐姿往往不正对，
        # 导致光标系统性偏移。recenter() 把当前姿态设为新零点。
        self.oyaw, self.opitch = 0.0, 0.0
        self._last_pose = None

    def recenter(self):
        """把当前头部姿态设为零点（解决自然坐姿不正对导致的偏移）。

        同时把光标移回屏幕中心——重置后当前姿态即对应中心。
        """
        if self._last_pose is not None:
            self.oyaw = self._last_pose[0]
            self.opitch = self._last_pose[1]
        self.px, self.py = 0.5, 0.5
        self._recent = []
        self.mouse.move_abs(0.5, 0.5)

    def ensure_engine(self):
        if self.engine is None:
            self.engine = HeadPoseEngine()
        return self.engine

    def close(self):
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None

    def reset(self):
        self._recent = []

    def process(self, rgb, mirror=True):
        """处理一帧：检测 -> 映射 -> 滤波 -> 移动鼠标。

        返回 info 字典（face/valid/ratio/point/face_pts），供标注与状态栏。
        """
        info = {"face": False, "valid": False,
                "ratio": None, "point": (self.px, self.py), "face_pts": []}
        try:
            engine = self.ensure_engine()
        except Exception:
            return info
        try:
            gp, ginfo = engine.detect(rgb, mirror=mirror)
        except Exception:
            return info
        info["face"] = bool(ginfo.get("face"))
        info["ratio"] = gp
        info["face_pts"] = ginfo.get("face_pts", [])
        if gp is None:
            self._recent = []
            return info
        self._last_pose = gp

        # 零点校正：当前姿态减去零点偏移（recenter 设定）
        ay = gp[0] - self.oyaw
        ap = gp[1] - self.opitch

        # 映射：标定系数优先，否则线性+增益（角度域：0=正对）
        coeffs = self.settings.get("head_calib")
        if coeffs and len(coeffs) == 12:
            sx, sy = PoseCalibrator.predict(coeffs, ay, ap)
        else:
            gain = float(self.settings.get("head_sensitivity", 1.0))
            sx = min(1.0, max(0.0, 0.5 + ay * gain))
            sy = min(1.0, max(0.0, 0.5 + ap * gain))

        # 中值聚合：窗口内取中位数，滤掉单帧野值
        self._recent.append((sx, sy))
        if len(self._recent) > self.MEDIAN_WIN:
            self._recent.pop(0)
        xs = sorted(p[0] for p in self._recent)
        ys = sorted(p[1] for p in self._recent)
        mx = xs[len(xs) // 2]
        my = ys[len(ys) // 2]

        # EMA 平滑 + 死区
        # 死区作用于"目标与当前位置的偏差"（而非单帧 EMA 增量）：
        # 头部缓慢转动时每帧增量极小，若按增量判死区会永远卡住不动
        alpha = float(self.settings.get("smoothing", 0.35))
        alpha = max(0.05, min(0.9, alpha))
        if abs(mx - self.px) > self.DEAD_ZONE or abs(my - self.py) > self.DEAD_ZONE:
            self.px += alpha * (mx - self.px)
            self.py += alpha * (my - self.py)
            self.mouse.move_abs(self.px, self.py)
        info["valid"] = True
        info["point"] = (self.px, self.py)
        return info


# ---------------------------------------------------------------------------
# 引擎工厂
# ---------------------------------------------------------------------------
def make_engine(kind="auto"):
    """kind: auto / mediapipe / skin。返回 (engine, 提示信息)。"""
    if kind == "auto":
        try:
            return MediaPipeEngine(), "自动选择：MediaPipe 引擎"
        except Exception:
            # 记录回退原因（打包版无控制台，便于诊断）
            try:
                with open(os.path.join(BASE_DIR, "engine_fallback.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(traceback.format_exc())
            except Exception:
                pass
            return SkinEngine(), "自动选择：肤色识别引擎（MediaPipe 初始化失败，已回退）"
    if kind == "mediapipe":
        return MediaPipeEngine(), "MediaPipe 引擎"
    if kind == "skin":
        return SkinEngine(), "肤色识别引擎"
    raise ValueError("未知引擎: %s" % kind)


def mediapipe_available():
    try:
        import mediapipe  # noqa: F401
        return os.path.isfile(os.path.join(BASE_DIR,
                                           MediaPipeEngine.MODEL_NAME))
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# 手势解释器：手势 -> 鼠标动作（含平滑 / 防抖 / 冷却）
# ---------------------------------------------------------------------------
class GestureInterpreter:
    HOLD_TIME = 0.10       # 手势需保持该时长才生效（防抖）
    CLICK_COOLDOWN = 0.7   # 两次单击/右键手势的最小间隔（秒）
    DCLICK_COOLDOWN = 0.4  # 双击手势的独立冷却（比单击短，支持连续双击）
    DRAG_HOLD_TIME = 0.45  # 拇指+食指捏合超过此时长 -> 进入拖拽（长按）
    MIN_TAP_TIME = 0.06    # 短于此的触摸视为抖动，不触发单击
    TAP_COOLDOWN = 0.06    # 连点最小间隔（快速两连捏=系统双击，不能太高）
    TOUCH_AFTERGLOW = 0.45  # 触摸手势结束后，4 指滚轮判定的冷静期（秒）

    def __init__(self, mouse, settings=None, on_event=None):
        self.mouse = mouse
        self.settings = settings if settings else dict(DEFAULT_SETTINGS)
        self.on_event = on_event          # 回调（用于状态栏展示），参数: (gesture, x, y)
        self.px, self.py = 0.5, 0.5
        self.lpx, self.lpy = 0.5, 0.5     # 双手模式：左手独立平滑坐标
        self.cand = GESTURE_NONE
        self.cand_since = 0.0
        self.eff = GESTURE_NONE
        self.last_click = 0.0
        self.last_dclick = 0.0
        self.last_tap = 0.0
        self.pinch_state = None     # None / "touch" / "drag"（拇指+食指捏合）
        self.pinch_start = 0.0
        self.pinch_miss = 0         # 连续未检测到触摸的帧数（释放防抖）
        self.touch_release_at = 0.0  # 最近一次触摸手势结束时刻（冷静期起点）
        self.prev_thumb_touch = 0   # 上一帧左手拇指触摸状态（检测松开沿）
        self.scroll_acc = 0.0
        self.scroll_dir = 0         # 4 指滚轮锁定方向：+1 上滚 / -1 下滚 / 0 未锁定
        self.last_t = time.perf_counter()

    def reset(self):
        self._exit_gesture(self.eff)
        if self.pinch_state == "drag":
            self.mouse.left_up()
        self.px, self.py = 0.5, 0.5
        self.lpx, self.lpy = 0.5, 0.5
        self.cand = GESTURE_NONE
        self.eff = GESTURE_NONE
        self.pinch_state = None
        self.pinch_miss = 0
        self.touch_release_at = 0.0
        self.prev_thumb_touch = 0
        self.scroll_acc = 0.0
        self.scroll_dir = 0
        self.last_dclick = 0.0

    # -- 分类（单手模式：任一主手） ------------------------------------------
    @staticmethod
    def classify(res):
        if not res.hand_found:
            return GESTURE_NONE
        # 握拳（手指全弯）时忽略拇指触摸，避免误判捏合拖拽/单击
        if res.pinch and sum(1 for x in res.fingers[1:] if x) == 0:
            return GESTURE_NONE
        if res.pinch:
            return GESTURE_DRAG
        f = res.fingers
        if f[0] and f[1] and f[2] and f[3] and f[4]:
            return GESTURE_DRAG
        if f[1] and f[2] and f[3] and f[4] and not f[0]:
            return GESTURE_SCROLL
        if f[1] and f[2] and f[3] and not f[4]:
            return GESTURE_RCLICK
        if f[1] and f[2] and not f[3] and not f[4]:
            return GESTURE_LCLICK
        if f[1] and not f[2] and not f[3] and not f[4]:
            return GESTURE_MOVE
        return GESTURE_NONE

    # -- 分类（双手模式：左手 = 动作手势）-------------------------------------
    @staticmethod
    def classify_left(h, game_mode=False):
        """返回 (gesture, index_pinch)。

        左手仅保留捏合（拇指触摸）判定：
        - 拇指+食指：单击 / 长按拖拽（index_pinch 状态机）
        - 拇指+中指：右键
        - 拇指+无名指：上滑
        - 拇指+小指：下滑
        游戏模式额外映射：五指张开 = Tab（锁定目标）。
        其余手指状态（1/2/3/4/5 指、握拳）一律无动作。
        握拳时拇指会压在手指上，需排除以免误判为捏合。
        """
        if h is None:
            return GESTURE_NONE, False
        t = h.thumb_touch
        extended = sum(1 for x in h.fingers[1:] if x)
        if t != 0 and extended == 0:
            return GESTURE_NONE, False      # 握拳（全弯）：忽略拇指触摸
        if t == 1:
            return GESTURE_NONE, True       # 拇指+食指：单击 / 长按拖拽
        if t == 2:
            return GESTURE_RCLICK, False    # 拇指+中指：右键
        if t == 3:
            return GESTURE_SCROLLUP, False  # 拇指+无名指：上滑
        if t == 4:
            return GESTURE_SCROLLDOWN, False  # 拇指+小指：下滑
        if game_mode and extended == 4 and h.fingers[0] is False:
            # 游戏模式：4 指（无拇指）= Tab 锁定目标
            return GESTURE_KEY_TAB, False
        return GESTURE_NONE, False

    # -- 主循环 --------------------------------------------------------------
    def update(self, res, dt=None, allow_move=True):
        """allow_move=False 时只处理手势动作（点击/滚轮/拖拽按键），
        不移动光标——光标由独立的 HeadController 驱动（头动开启时）。"""
        if self.settings.get("hands_mode", "single") == "dual":
            return self._update_dual(res, dt, allow_move)
        return self._update_single(res, dt, allow_move)

    def _smooth_pointer(self, pointer):
        alpha = float(self.settings.get("smoothing", 0.35))
        alpha = max(0.02, min(0.95, alpha))
        self.px += alpha * (pointer[0] - self.px)
        self.py += alpha * (pointer[1] - self.py)

    def _smooth_left_pointer(self, pointer):
        """双手模式：左手位置的独立平滑（用于滚轮方向）。"""
        alpha = float(self.settings.get("smoothing", 0.35))
        alpha = max(0.02, min(0.95, alpha))
        self.lpx += alpha * (pointer[0] - self.lpx)
        self.lpy += alpha * (pointer[1] - self.lpy)

    def _update_single(self, res, dt=None, allow_move=True):
        now = time.perf_counter()
        if dt is None:
            dt = max(now - self.last_t, 1e-4)
        self.last_t = now

        gesture = self.classify(res)
        self._smooth_pointer(res.pointer)

        # 手势防抖：同一手势保持 HOLD_TIME 才切换生效
        # 4 指滚轮进入更谨慎：张开手掌时手指外展抖动易把 5 指误认成 4 指
        if gesture != self.cand:
            self.cand = gesture
            self.cand_since = now
        hold = 0.20 if gesture == GESTURE_SCROLL else self.HOLD_TIME
        if now - self.cand_since >= hold:
            if self.cand != self.eff:
                self._exit_gesture(self.eff)
                self.eff = self.cand
                self._enter_gesture(self.eff, now)

        eff = self.eff

        # 执行持续动作（头动开启时 allow_move=False，光标由 HeadController 管）
        if allow_move and eff in (GESTURE_MOVE, GESTURE_DRAG):
            self._do_move(dt)
        elif eff == GESTURE_SCROLL:
            self._do_scroll_locked(dt, raw_y=res.pointer[1])

        if self.on_event:
            self.on_event(eff, self.px, self.py)
        return eff

    def _update_dual(self, res, dt=None, allow_move=True):
        """双手模式：右手（用户视角）移动光标，左手手势触发动作。

        头动开启时 allow_move=False：本方法只处理左手动作，
        光标移动由独立的 HeadController 负责（两套逻辑完全解耦）。"""
        now = time.perf_counter()
        if dt is None:
            dt = max(now - self.last_t, 1e-4)
        self.last_t = now

        right = res.hand("right")
        left = res.hand("left")

        # 检测拇指触摸松开沿（任何触摸手势结束都记录，用于滚轮冷静期）
        cur_touch = left.thumb_touch if left is not None else 0
        if cur_touch == 0 and self.prev_thumb_touch != 0:
            self.touch_release_at = now
        self.prev_thumb_touch = cur_touch

        # 右手在场：平滑跟随食指尖
        if right is not None:
            self._smooth_pointer(right.pointer)
        # 左手在场：独立平滑（滚轮方向用）
        if left is not None:
            self._smooth_left_pointer(left.pointer)

        # 左手手势 -> 动作（返回 (gesture, index_pinch)）
        gesture, index_pinch = self.classify_left(
            left, game_mode=bool(self.settings.get("game_mode")))

        # 触摸手势结束后的冷静期：手指放松时容易自然摆出 4 指姿态，
        # 若手恰在下半屏会被误判成"下滚"导致停不下来——冷静期内
        # 暂停 4 指滚轮判定（点击类手势不受影响）
        if gesture == GESTURE_SCROLL \
                and now - self.touch_release_at < self.TOUCH_AFTERGLOW:
            gesture = GESTURE_NONE

        # 拇指+食指捏合：单击 / 长按拖拽 独立状态机（优先级最高）
        self._update_index_pinch(index_pinch, now)

        # 普通手势防抖：同一手势保持 HOLD_TIME 才切换生效
        # 4 指滚轮进入更谨慎：张开手掌时手指外展抖动易把 5 指误认成 4 指
        if gesture != self.cand:
            self.cand = gesture
            self.cand_since = now
        hold = 0.20 if gesture == GESTURE_SCROLL else self.HOLD_TIME
        if now - self.cand_since >= hold:
            if self.cand != self.eff:
                self._exit_gesture(self.eff)
                self.eff = self.cand
                self._enter_gesture(self.eff, now)

        eff = self.eff

        # 指针可用时持续移动光标（头动开启时 allow_move=False，不抢光标）
        if allow_move and right is not None:
            self._do_move(dt)
        if eff == GESTURE_SCROLL:
            self._do_scroll_locked(dt, use_left=(left is not None),
                                   raw_y=(left.pointer[1] if left is not None else None))
        elif eff == GESTURE_SCROLLUP:
            self._do_scroll_fixed(dt, +1)
        elif eff == GESTURE_SCROLLDOWN:
            self._do_scroll_fixed(dt, -1)

        if self.on_event:
            self.on_event(eff, self.px, self.py)
        return eff

    # -- 拇指+食指：单击 / 长按拖拽 ------------------------------------------
    PINCH_RELEASE_FRAMES = 2   # 连续多少帧未检测到触摸才确认"松开"（防抖动闪断）

    def _update_index_pinch(self, pinching, now):
        if pinching:
            self.pinch_miss = 0
            if self.pinch_state is None:
                self.pinch_state = "touch"
                self.pinch_start = now
            elif self.pinch_state == "touch" \
                    and now - self.pinch_start >= self.DRAG_HOLD_TIME:
                self.pinch_state = "drag"
                self.mouse.left_down()      # 长按 -> 进入拖拽
        else:
            if self.pinch_state is None:
                self.pinch_miss = 0
                return
            self.pinch_miss += 1
            if self.pinch_miss < self.PINCH_RELEASE_FRAMES:
                return                      # 闪断，视为仍在触摸
            if self.pinch_state == "touch":
                held = now - self.pinch_start
                # 松开：短触摸 = 单击
                if held >= self.MIN_TAP_TIME \
                        and now - self.last_tap >= self.TAP_COOLDOWN:
                    self.mouse.click_left()
                    self.last_tap = now
                    self.last_click = now   # 防止随后手势状态机重复点击
                self.pinch_state = None
                self.touch_release_at = now
            elif self.pinch_state == "drag":
                self.mouse.left_up()        # 长按结束 -> 释放
                self.pinch_state = None
                self.touch_release_at = now
            self.pinch_miss = 0

    # -- 手势切换 ------------------------------------------------------------
    def _enter_gesture(self, g, now):
        if g == GESTURE_LCLICK and now - self.last_click >= self.CLICK_COOLDOWN:
            self.mouse.click_left()
            self.last_click = now
        elif g == GESTURE_RCLICK and now - self.last_click >= self.CLICK_COOLDOWN:
            self.mouse.click_right()
            self.last_click = now
        elif g == GESTURE_DCLICK and now - self.last_dclick >= self.DCLICK_COOLDOWN:
            self.mouse.double_click()
            self.last_dclick = now
            self.last_click = now   # 防双击后手指姿态变化误连击单击
        elif g == GESTURE_DRAG:
            self.mouse.left_down()
        elif g == GESTURE_KEY_TAB and now - self.last_click >= self.CLICK_COOLDOWN:
            self.mouse.key_tap(MouseController.VK_TAB)
            self.last_click = now

    def _exit_gesture(self, g):
        if g == GESTURE_DRAG:
            self.mouse.left_up()
        # 手势退出时丢弃未滚完的累积并解除方向锁定，避免跨手势残留
        self.scroll_acc = 0.0
        self.scroll_dir = 0

    # -- 持续动作 ------------------------------------------------------------
    def _do_move(self, dt):
        sens = float(self.settings.get("sensitivity", 1.0))
        mode = self.settings.get("mode", "absolute")
        if mode == "relative":
            dx = self.px - 0.5
            dy = self.py - 0.5
            dead = 0.07
            mag = math.hypot(dx, dy)
            if mag > dead:
                k = (mag - dead) ** 1.6 / ((1.0 - dead) ** 1.6)
                speed = 2600.0 * sens * k * dt
                if mag > 0:
                    self.mouse.move_rel(dx / mag * speed, dy / mag * speed)
        else:
            nx = 0.5 + (self.px - 0.5) * sens
            ny = 0.5 + (self.py - 0.5) * sens
            self.mouse.move_abs(nx, ny)

    def _do_scroll_locked(self, dt, use_left=False, raw_y=None):
        """4 指滚轮：手势进入瞬间锁定方向，之后固定速率滚动。

        方向以进入时刻的原始手位置为准（上半屏=向上滚 / 下半屏=向下滚），
        不使用平滑坐标——EMA 平滑有惯性，手势刚做出时平滑值还停留在
        几帧前的旧位置，会导致方向锁反。期间手的位置变化不影响方向。
        收起手势重新做出即可切换方向。
        """
        if self.scroll_dir == 0:
            if raw_y is not None:
                py = raw_y
            else:
                py = self.lpy if use_left else self.py
            self.scroll_dir = +1 if py < 0.5 else -1
        self._do_scroll_fixed(dt, self.scroll_dir)

    def _do_scroll_fixed(self, dt, direction):
        """固定方向持续滚动（拇指+无名指=上滑 / 拇指+小指=下滑 / 4 指锁定滚轮）。

        速度由独立的「滚轮速度」设置控制（格/秒）。
        """
        rate_mult = float(self.settings.get("scroll_speed", 1.0))
        rate = direction * 6.0 * rate_mult    # 格/秒
        self.scroll_acc += rate * dt
        if abs(self.scroll_acc) >= 0.5:
            clicks = int(self.scroll_acc / 0.5)
            self.scroll_acc -= clicks * 0.5
            self.mouse.scroll(clicks)


# ---------------------------------------------------------------------------
# 摄像头工作线程：采集 -> 识别 -> 控制 -> 出帧
# ---------------------------------------------------------------------------
class CameraWorker(threading.Thread):
    FRAME_W, FRAME_H = 640, 480
    MIN_PROC_INTERVAL = 0.015  # 处理上限 ~66fps（及时消费帧，减少积压延迟）
    HAND_INTERVAL = 3          # 头动开启时手部检测降频间隔（帧）

    def __init__(self, settings, on_frame=None, on_error=None, dry_run=False):
        super().__init__(daemon=True)
        self.settings = settings
        self.on_frame = on_frame   # 回调: (bgr_frame, info_dict) 供 GUI 渲染
        self.on_error = on_error
        self.dry_run = dry_run
        self.stop_event = threading.Event()
        self.started = threading.Event()
        self._lock = threading.Lock()
        self.frame = None
        self.info = {}
        self._frame_idx = 0
        self._cached_hand_res = None   # 手部检测降频时复用上一帧结果

    # GUI 修改设置后调用
    def apply_settings(self, new_settings):
        with self._lock:
            self.settings.update(new_settings)

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        cap = None
        engine = None
        head_ctrl = None
        interp = None
        mouse = DryRunMouse() if self.dry_run else MouseController()
        try:
            engine, _ = make_engine(self.settings.get("engine", "auto"))
            interp = GestureInterpreter(mouse, self.settings,
                                        on_event=self._on_gesture)
            cap = _open_camera(self.settings.get("camera", "auto"),
                               stop_check=self.stop_event.is_set)
            if cap is None:
                if not self.stop_event.is_set():
                    self.on_error("未找到可用摄像头。请检查摄像头是否被占用，"
                                  "或在界面中手动指定摄像头序号。")
                self.started.set()
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
            cap.set(cv2.CAP_PROP_FPS, 60)   # 请求 60fps，硬件不支持时自动回落
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            fps_ema, frame_count, last_sec = 0.0, 0, time.monotonic()
            last_proc = time.perf_counter()
            self.started.set()

            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                now = time.perf_counter()
                # 节流：距上次处理不足 MIN_PROC_INTERVAL 则跳过本帧
                # （降低 CPU 占用，后台运行时识别与鼠标控制依然流畅）
                if now - last_proc < self.MIN_PROC_INTERVAL:
                    continue
                dt = min(now - last_proc, 0.2)
                last_proc = now

                with self._lock:
                    mirror = bool(self.settings.get("mirror", True))
                    show_lm = bool(self.settings.get("show_landmarks", True))
                    show_hint = bool(self.settings.get("show_hint", True))
                    cur = dict(self.settings)

                if mirror:
                    frame = cv2.flip(frame, 1)

                touch_thr = float(cur.get("touch_sensitivity", 0.55))
                head_on = bool(cur.get("head_enabled"))
                rgb = None

                # 头动开启时：手部检测降频（动作手势不需要每帧），
                # 人脸检测每帧全速 -> 头动刷新率高、延迟低
                hand_skip = head_on \
                    and (self._frame_idx % self.HAND_INTERVAL != 0) \
                    and self._cached_hand_res is not None
                if hand_skip:
                    res = self._cached_hand_res
                else:
                    if engine.name == "MediaPipe":
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = engine.detect(rgb, mirror=mirror,
                                            touch_threshold=touch_thr)
                    else:
                        res = engine.detect(frame, mirror=mirror)
                    self._cached_hand_res = res
                self._frame_idx += 1

                # 头动光标（独立逻辑）：head_enabled 时由 HeadController
                # 全权负责"头指向哪光标去哪"，与手势解释器互不感知
                if head_on:
                    if head_ctrl is None:
                        head_ctrl = HeadController(cur, mouse)
                    # 重置中心请求（GUI 按钮触发）
                    if cur.get("head_recenter"):
                        head_ctrl.recenter()
                        with self._lock:
                            self.settings["head_recenter"] = False
                    if rgb is None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    ginfo = head_ctrl.process(rgb, mirror=mirror)
                    res.gaze_point = ginfo.get("ratio")
                    res.gaze_valid = bool(ginfo.get("valid"))
                    res.face_found = bool(ginfo.get("face"))
                    res.face_pts = ginfo.get("face_pts", [])
                    res.face_landmarks = ginfo.get("landmarks")
                else:
                    if head_ctrl is not None:
                        head_ctrl.close()
                        head_ctrl = None

                eff = GESTURE_NONE
                if not cur.get("calibrating"):
                    # 头动开启时解释器只做手势动作（点击/滚轮/拖拽按键），
                    # 光标移动交给 HeadController
                    eff = interp.update(res, dt, allow_move=(not head_on))

                # FPS
                frame_count += 1
                if now - last_sec >= 1.0:
                    fps_ema = frame_count / (now - last_sec)
                    frame_count, last_sec = 0, now

                # 画面渲染三种版本：
                #   raw    = 纯摄像头画面（只看动画）
                #   marker = 黑底 + 手部标记 + 手势文字（只看标记）
                #   full   = 画面 + 标记叠加（都看）
                hands_mode = cur.get("hands_mode", "single")
                frames = {
                    "raw": frame,
                    "marker": _annotate(np.zeros_like(frame), res, eff,
                                        fps_ema, engine.name, show_lm, show_hint,
                                        hands_mode=hands_mode, head_on=head_on),
                    "full": _annotate(frame.copy(), res, eff, fps_ema,
                                      engine.name, show_lm, show_hint,
                                      hands_mode=hands_mode, head_on=head_on),
                }

                info = {
                    "gesture": eff,
                    "hand": res.hand_found,
                    "fingers": res.count,
                    "fps": fps_ema,
                    "engine": engine.name,
                    "screen": (mouse.screen_w, mouse.screen_h),
                    "hands_mode": hands_mode,
                    "left": res.hand("left") is not None,
                    "right": res.hand("right") is not None,
                    "face": res.face_found,
                    "blink": False,
                    "gaze_valid": res.gaze_valid,
                    "gaze_ratio": res.gaze_point,   # 原始头部朝向比例（标定用）
                    "gaze_on": head_on,
                    "head_on": head_on,
                }
                payload = {"frames": frames, "info": info}
                with self._lock:
                    self.frame = frames["full"]
                    self.info = info
                if self.on_frame:
                    self.on_frame(payload)
        except Exception:
            traceback.print_exc()
            self.on_error("运行出错：\n" + traceback.format_exc(limit=3))
        finally:
            # 释放可能仍按住的鼠标按键（拖拽中停止/出错时不至于卡键）
            if interp is not None:
                try:
                    interp.reset()
                except Exception:
                    pass
            if head_ctrl is not None:
                try:
                    head_ctrl.close()
                except Exception:
                    pass
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass
            if cap is not None:
                cap.release()
            self.started.set()

    def _on_gesture(self, gesture, x, y):
        pass


def _open_camera(choice, stop_check=None):
    """choice: 'auto' 或数字字符串。返回 VideoCapture 或 None。

    stop_check: 可选回调，返回 True 时立即放弃（用于停止时快速取消）。
    """
    indexes = []
    if str(choice).lower() == "auto":
        indexes = list(range(4))
    else:
        try:
            indexes = [int(choice)]
        except (TypeError, ValueError):
            indexes = list(range(4))
    for idx in indexes:
        if stop_check is not None and stop_check():
            return None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
                cap.release()
            else:
                cap.release()
        except Exception:
            pass
    for idx in indexes:
        if stop_check is not None and stop_check():
            return None
        try:
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    return cap
                cap.release()
            else:
                cap.release()
        except Exception:
            pass
    return None


_HAND_CONNS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
               (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
               (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20),
               (0, 17)]

# 脸部特征点（MediaPipe FaceMesh 索引）：眉、眼、鼻、嘴、下巴、脸侧
FACE_FEATURE_IDX = [
    # 眉毛
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
    # 眼睛
    33, 160, 158, 133, 153, 144, 145, 159,
    263, 387, 385, 362, 380, 373, 374, 386,
    # 鼻
    1, 2, 4, 5, 6, 98, 327,
    # 嘴
    0, 17, 61, 291, 13, 14,
    # 下巴与脸侧
    152, 10, 234, 454, 58, 288, 172, 397, 365, 377,
]

# 脸部特征连线（多识别点可视化）
FACE_FEATURE_LINES = [
    (70, 63), (63, 105), (105, 66), (66, 107),            # 左眉
    (336, 296), (296, 334), (334, 293), (293, 300),       # 右眉
    (33, 160), (160, 158), (158, 133), (133, 153),        # 左眼环
    (153, 144), (144, 145), (145, 159), (159, 33),
    (263, 387), (387, 385), (385, 362), (362, 380),       # 右眼环
    (380, 373), (373, 374), (374, 386), (386, 263),
    (1, 2), (2, 4), (4, 5), (5, 6),                       # 鼻梁
    (98, 327),                                            # 鼻翼
    (61, 0), (0, 291), (61, 13), (13, 14), (14, 291),     # 嘴
    (10, 152), (152, 234),                                # 下巴
    (10, 58), (234, 288),                                 # 脸侧
]


def _annotate(frame, res, gesture, fps, engine_name, show_lm, show_hint,
              hands_mode="single", head_on=False):
    out = frame.copy()
    h, w = out.shape[:2]

    if hands_mode == "dual" and res.hands:
        # 双手模式：左手蓝、右手绿（即使只检测到一只手也按左右着色）
        color_map = {"left": (255, 160, 60), "right": (0, 255, 0)}
        for hand in res.hands:
            color = color_map.get(hand.label, (200, 200, 200))
            if res.engine_name == "MediaPipe" and hand.landmarks and show_lm:
                pts = [(int(l[0] * w), int(l[1] * h)) for l in hand.landmarks]
                for a, b in _HAND_CONNS:
                    cv2.line(out, pts[a], pts[b], color, 2)
                for p in pts:
                    cv2.circle(out, p, 4, color, -1)
            # 右手是光标手：画黄色目标圈（头动开启时无右手光标圈）
            if not head_on and hand.label == "right":
                px, py = int(hand.pointer[0] * w), int(hand.pointer[1] * h)
                cv2.circle(out, (px, py), 8, (0, 255, 255), -1)
                cv2.circle(out, (px, py), 14, (0, 255, 255), 2)
    else:
        # 单手模式 / 只检测到一只手
        if res.hand_found and not head_on:
            px, py = int(res.pointer[0] * w), int(res.pointer[1] * h)
            cv2.circle(out, (px, py), 8, (0, 255, 255), -1)
            cv2.circle(out, (px, py), 14, (0, 255, 255), 2)
        if res.engine_name == "MediaPipe" and res.landmarks and show_lm:
            pts = [(int(l[0] * w), int(l[1] * h)) for l in res.landmarks]
            for a, b in _HAND_CONNS:
                cv2.line(out, pts[a], pts[b], (0, 255, 0), 2)
            for p in pts:
                cv2.circle(out, p, 4, (0, 0, 255), -1)

    if head_on and show_lm:
        # 头动开启：画多特征点网格（眉/眼/鼻/嘴/下巴/脸侧）
        if res.face_landmarks:
            pts = [(int(x * w), int(y * h)) for x, y in res.face_landmarks]
            for a, b in FACE_FEATURE_LINES:
                cv2.line(out, pts[a], pts[b], (0, 200, 0), 1)
            for idx in FACE_FEATURE_IDX:
                cv2.circle(out, pts[idx], 2, (0, 255, 0), -1)
        # 眼角连线 + 鼻尖 + 姿态十字
        if res.face_pts:
            (lx, ly), (rx, ry), (mx, my), (nx, ny) = res.face_pts
            a = (int(lx * w), int(ly * h))
            b = (int(rx * w), int(ry * h))
            n = (int(nx * w), int(ny * h))
            cv2.line(out, a, b, (0, 255, 255), 2)
            cv2.circle(out, n, 6, (0, 255, 255), -1)
            cv2.line(out, n, (int(mx * w), int(my * h)), (200, 200, 0), 1)
        if res.gaze_valid and res.gaze_point is not None:
            gx, gy = int(res.gaze_point[0] * w), int(res.gaze_point[1] * h)
            cv2.circle(out, (gx, gy), 10, (0, 255, 128), 2)
            cv2.line(out, (gx - 14, gy), (gx + 14, gy), (0, 255, 128), 1)
            cv2.line(out, (gx, gy - 14), (gx, gy + 14), (0, 255, 128), 1)

    if show_hint:
        label = "Gesture: %s" % _gesture_en(res, gesture)
        left = res.hand("left")
        right = res.hand("right")
        if hands_mode == "dual":
            l_txt = ("L:%d" % left.count) if left is not None else "L:--"
            if left is not None and left.thumb_touch:
                touch_names = {1: "IDX", 2: "MID", 3: "RNG", 4: "PNK"}
                l_txt += "+%s" % touch_names.get(left.thumb_touch, "?")
            r_txt = ("R:%d" % right.count) if right is not None else "R:--"
            label = "%s  [%s %s]" % (label, l_txt, r_txt)
        if head_on:
            face_txt = ("FACE:%s" % ("OK" if res.face_found else "--"))
            label = "%s  [%s]" % (label, face_txt)
        cv2.rectangle(out, (8, 8), (w - 8, 40), (30, 30, 30), -1)
        cv2.putText(out, label, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 128), 2, cv2.LINE_AA)
        mode_txt = "DUAL" if hands_mode == "dual" else "SINGLE"
        mode_txt += "+HEAD" if head_on else ""
        fps_txt = "FPS %.1f | %s | %s" % (fps, engine_name, mode_txt)
        cv2.putText(out, fps_txt, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _gesture_en(res, gesture):
    if not res.hand_found:
        return "NO HAND (mouse frozen)"
    return {
        GESTURE_MOVE: "MOVE",
        GESTURE_LCLICK: "LEFT CLICK",
        GESTURE_RCLICK: "RIGHT CLICK",
        GESTURE_DCLICK: "DOUBLE CLICK",
        GESTURE_SCROLL: "SCROLL",
        GESTURE_SCROLLUP: "SCROLL UP",
        GESTURE_SCROLLDOWN: "SCROLL DOWN",
        GESTURE_DRAG: "DRAG",
        GESTURE_NONE: "IDLE",
    }.get(gesture, "?")


# ---------------------------------------------------------------------------
# 设置读写
# ---------------------------------------------------------------------------
def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            s.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
    except Exception:
        pass
    # 旧版设置迁移：
    # 1) hands_mode=="gaze"（眼动曾作为控制模式）-> 双手 + 头动开关
    # 2) 旧眼动键 -> 头动键
    # 3) 三点比例法时代的标定系数作废（已升级为矩阵角度版），重置灵敏度
    if s.get("hands_mode") == "gaze":
        s["hands_mode"] = "dual"
        s["head_enabled"] = True
    for old, new in (("gaze_enabled", "head_enabled"),
                     ("gaze_sensitivity", "head_sensitivity"),
                     ("gaze_calib", "head_calib")):
        if old in s and s.get(new) is None:
            s[new] = s[old]
    if s.get("head_calib") and not s.get("head_pose_v2"):
        s["head_calib"] = None
        s["head_sensitivity"] = 1.0
        s["head_pose_v2"] = True
    return s


def save_settings(s):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 图形界面
# ---------------------------------------------------------------------------
def build_gui(dry_run=False):
    import tkinter as tk
    from tkinter import ttk, messagebox

    try:
        from PIL import Image, ImageTk
    except ImportError:
        Image = ImageTk = None

    class MonitorWindow(tk.Toplevel):
        """屏幕右下角悬浮监视窗：置顶、可自由拖动与调整大小。

        显示模式：动画（纯画面）/ 标记（黑底骨架）/ 都看（画面+标记），
        并大字显示当前动作与具体手势说明。
        """

        def __init__(self, master, settings):
            super().__init__(master)
            self.title("手势监视（可拖动缩放）")
            self.attributes("-topmost", True)
            w = max(120, int(settings.get("monitor_w", 320)))
            h = max(90, int(settings.get("monitor_h", 240)))
            x = max(0, self.winfo_screenwidth() - w - 24)
            y = max(0, self.winfo_screenheight() - h - 64)
            self.geometry("%dx%d+%d+%d" % (w, h, x, y))
            self.minsize(120, 90)
            self.configure(bg="#101018")

            # 顶部：显示模式三选
            top = ttk.Frame(self)
            top.pack(fill="x", padx=4, pady=(4, 0))
            self.mode_var = tk.StringVar(
                value=str(settings.get("monitor_mode", "full")))
            for text, val in (("动画", "raw"), ("标记", "marker"), ("都看", "full")):
                ttk.Radiobutton(top, text=text, value=val,
                                variable=self.mode_var).pack(side="left")

            # 动作大字 + 说明
            self.gesture_label = tk.Label(
                self, text="无手 · 鼠标已冻结",
                font=("Microsoft YaHei UI", 13, "bold"),
                fg="#9e9e9e", bg="#101018", anchor="w")
            self.gesture_label.pack(fill="x", padx=6)
            self.hint_label = tk.Label(
                self, text="手伸到摄像头前恢复控制",
                font=("Microsoft YaHei UI", 8),
                fg="#888899", bg="#101018", anchor="w", justify="left")
            self.hint_label.pack(fill="x", padx=6, pady=(0, 2))

            # 画面
            self.img_label = tk.Label(self, bg="#000000")
            self.img_label.pack(fill="both", expand=True, padx=4, pady=4)

            self.protocol("WM_DELETE_WINDOW", self._on_close)

        def set_gesture(self, info):
            g = info.get("gesture", GESTURE_NONE)
            hands_mode = info.get("hands_mode", "dual")
            if info.get("head_on"):
                if not info.get("face"):
                    self.gesture_label.config(text="未检测到人脸 · 光标已冻结",
                                              fg="#9e9e9e")
                    self.hint_label.config(text="请正对摄像头，保持光线充足")
                else:
                    self.gesture_label.config(
                        text=GESTURE_LABEL_CN.get(g, "?"),
                        fg=GESTURE_COLOR_CN.get(g, "#9e9e9e"))
                    self.hint_label.config(text="头部朝向=移动 · " + GESTURE_HINT_CN.get(g, ""))
            elif not info.get("hand"):
                self.gesture_label.config(text="无手 · 鼠标已冻结", fg="#9e9e9e")
                self.hint_label.config(text="手伸到摄像头前恢复控制")
            else:
                self.gesture_label.config(
                    text=GESTURE_LABEL_CN.get(g, "?"),
                    fg=GESTURE_COLOR_CN.get(g, "#9e9e9e"))
                self.hint_label.config(text=GESTURE_HINT_CN.get(g, ""))

        def _on_close(self):
            try:
                if self.master is not None:
                    self.master._close_monitor()
                else:
                    self.destroy()
            except Exception:
                try:
                    self.destroy()
                except Exception:
                    pass

    class CalibrationWindow(tk.Toplevel):
        """全屏九点头动标定：依次显示 9 个圆点，用户头部转向圆点方向，
        每点采集头部朝向比例样本，完成后岭回归拟合个人映射。"""

        POINTS = [(0.5, 0.5), (0.12, 0.12), (0.88, 0.12), (0.88, 0.88),
                  (0.12, 0.88), (0.5, 0.12), (0.88, 0.5), (0.5, 0.88),
                  (0.12, 0.5)]
        DWELL = 1.4        # 每点标准停留秒数
        TRANSITION = 0.5   # 每点前段过渡时间：头部还在转过去，丢弃不采样
        MAX_DWELL = 3.5    # 样本不足时自动延长的上限
        MIN_SAMPLES = 4    # 每点最少样本帧数

        def __init__(self, master, on_done):
            super().__init__(master)
            self.on_done = on_done
            try:
                self.attributes("-fullscreen", True)
                self.attributes("-topmost", True)
            except Exception:
                pass
            self.configure(bg="#000000")
            self.canvas = tk.Canvas(self, bg="#000000", highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)
            self.info_label = tk.Label(
                self, text="头动标定：请把头转向屏幕上的蓝色圆点方向（Esc 取消）",
                font=("Microsoft YaHei UI", 16), bg="#000000", fg="#ffffff")
            self.info_label.pack(pady=16)
            self.samples = []
            self._cur = -1
            self._dwell_start = 0.0
            self._buf = []
            self._aborted = False
            self.protocol("WM_DELETE_WINDOW", self._abort)
            self.bind("<Escape>", lambda e: self._abort())
            self.after(50, self._tick)

        def _tick(self):
            if self._aborted:
                return
            # 采样：从主窗口帧队列取最新头部朝向比例
            # （过渡期内丢弃——头部还在转向圆点，数据不可靠）
            elapsed = 0.0
            if self._cur >= 0:
                elapsed = time.monotonic() - self._dwell_start
            try:
                while True:
                    payload = self.master.frame_queue.get_nowait()
                    if self._cur >= 0 and elapsed >= self.TRANSITION \
                            and payload["info"].get("gaze_valid") \
                            and payload["info"].get("gaze_ratio"):
                        self._buf.append(payload["info"]["gaze_ratio"])
            except queue.Empty:
                pass
            if self._cur < 0:
                self._next_point()
            elif elapsed >= self.DWELL:
                # 样本不足自动延长停留，直到上限
                if len(self._buf) >= self.MIN_SAMPLES \
                        or elapsed >= self.MAX_DWELL:
                    self._collect()
                    if self._cur + 1 < len(self.POINTS):
                        self._next_point()
                    else:
                        self._finish()
            self.after(50, self._tick)

        def _next_point(self):
            self._cur += 1
            sx, sy = self.POINTS[self._cur]
            self.canvas.delete("all")
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w < 100:
                w, h = self.winfo_screenwidth(), self.winfo_screenheight()
            x, y = int(sx * w), int(sy * h)
            r = 22
            self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                    fill="#00c8ff", outline="#ffffff", width=3)
            self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="#ffffff")
            self.info_label.config(
                text="标定点 %d/%d —— 头部转向蓝色圆点方向，保持片刻"
                % (self._cur + 1, len(self.POINTS)))
            self._dwell_start = time.monotonic()
            self._buf = []

        def _collect(self):
            if len(self._buf) < self.MIN_SAMPLES:
                return
            xs = sorted(b[0] for b in self._buf)
            ys = sorted(b[1] for b in self._buf)
            rx = xs[len(xs) // 2]
            ry = ys[len(ys) // 2]
            sx, sy = self.POINTS[self._cur]
            self.samples.append((rx, ry, sx, sy))

        def _finish(self):
            coeffs = PoseCalibrator.fit(self.samples)
            err = None
            if coeffs is not None:
                errs = []
                for rx, ry, sx, sy in self.samples:
                    px, py = PoseCalibrator.predict(coeffs, rx, ry)
                    errs.append(math.hypot(px - sx, py - sy))
                if errs:
                    err = sum(errs) / len(errs)
            self._aborted = True
            try:
                self.destroy()
            except Exception:
                pass
            self.on_done(coeffs, len(self.samples), err)

        def _abort(self):
            if self._aborted:
                return
            self._aborted = True
            try:
                self.destroy()
            except Exception:
                pass
            self.on_done(None, len(self.samples))

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.dry_run = dry_run
            self.title("%s v%s" % (APP_NAME, VERSION))
            self.geometry("1000x640")
            self.minsize(920, 600)
            self.settings = load_settings()
            self.worker = None
            self.monitor = None
            self.frame_queue = queue.Queue(maxsize=2)
            self._closed = False
            self._stat = {"gesture": GESTURE_NONE, "fps": 0.0,
                          "engine": "-", "hand": False}

            self._style()
            self._build_widgets()
            self.protocol("WM_DELETE_WINDOW", self._on_close)
            # 按设置自动打开右下角监视窗
            if self.monitor_var.get():
                self._open_monitor()
            self.after(20, self._poll)   # 50fps 上限渲染轮询

        # -- 界面 ------------------------------------------------------------
        def _style(self):
            self.configure(bg="#1e1e2e")
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure(".", background="#1e1e2e", foreground="#e0e0e0",
                            fieldbackground="#2a2a3c")
            style.configure("TFrame", background="#1e1e2e")
            style.configure("TLabelframe", background="#252538",
                            foreground="#c0c0d0")
            style.configure("TLabelframe.Label", background="#252538",
                            foreground="#9fd0ff")
            style.configure("TLabel", background="#1e1e2e", foreground="#e0e0e0")
            style.configure("Title.TLabel", font=("Microsoft YaHei UI", 13, "bold"),
                            foreground="#ffffff")
            style.configure("Accent.TButton", font=("Microsoft YaHei UI", 11, "bold"))
            style.map("Accent.TButton",
                      background=[("active", "#2f7a4f")])

        def _build_widgets(self):
            left = ttk.Frame(self)
            left.pack(side="left", fill="both", expand=True, padx=(10, 6), pady=10)

            self.video_label = tk.Label(left, bg="#000000", text="摄像头画面\n点击「开始」后显示",
                                        fg="#808080", font=("Microsoft YaHei UI", 12))
            self.video_label.pack(fill="both", expand=True)

            self.status_bar = tk.Label(left, text="状态：未启动    |    引擎：-    |    FPS：-    |    手势：-",
                                       anchor="w", bg="#252538", fg="#9fd0ff",
                                       font=("Microsoft YaHei UI", 10))
            self.status_bar.pack(fill="x", pady=(6, 0))

            right = ttk.Frame(self, width=330)
            right.pack(side="right", fill="y", padx=(6, 10), pady=10)
            right.pack_propagate(False)

            ttk.Label(right, text="手势鼠标控制台", style="Title.TLabel").pack(anchor="w")

            # 启动
            self.btn_start = ttk.Button(right, text="▶  开 始 识 别",
                                        style="Accent.TButton",
                                        command=self._toggle_start)
            self.btn_start.pack(fill="x", pady=(8, 4))

            self.btn_calib = ttk.Button(right, text="🧭  头 动 标 定",
                                        command=self._start_calibration)
            self.btn_calib.pack(fill="x", pady=(0, 4))

            self.btn_recenter = ttk.Button(right, text="⌖  重置中心（当前姿态设为正中）",
                                           command=self._recenter_head)
            self.btn_recenter.pack(fill="x", pady=(0, 4))

            # 设备
            box = ttk.Labelframe(right, text=" 设备 ")
            box.pack(fill="x", pady=4)
            ttk.Label(box, text="摄像头").grid(row=0, column=0, sticky="w", padx=6, pady=3)
            self.cam_var = tk.StringVar(value=str(self.settings.get("camera", "auto")))
            cam_choices = ["auto", "0", "1", "2", "3"]
            ttk.Combobox(box, textvariable=self.cam_var, values=cam_choices,
                         width=6, state="readonly").grid(row=0, column=1, sticky="w")
            ttk.Label(box, text="识别引擎").grid(row=1, column=0, sticky="w", padx=6, pady=3)
            self.eng_var = tk.StringVar(value=str(self.settings.get("engine", "auto")))
            eng_choices = ["auto", "mediapipe", "skin"]
            if not mediapipe_available():
                eng_choices = ["auto", "skin"]
            self.eng_combo = ttk.Combobox(box, textvariable=self.eng_var,
                                          values=eng_choices, width=12, state="readonly")
            self.eng_combo.grid(row=1, column=1, sticky="w")
            self.engine_note = ttk.Label(box, text="", foreground="#88c0ff",
                                         wraplength=270, justify="left")
            self.engine_note.grid(row=2, column=0, columnspan=2, sticky="w",
                                  padx=6, pady=(2, 6))
            self._update_engine_note()

            # 参数
            box2 = ttk.Labelframe(right, text=" 参数 ")
            box2.pack(fill="x", pady=4)

            # 控制模式
            self.hands_var = tk.StringVar(value=str(self.settings.get("hands_mode", "dual")))
            ttk.Label(box2, text="控制模式（手势）", foreground="#9fd0ff").grid(
                row=0, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
            self.rb_dual = ttk.Radiobutton(box2, text="双手模式（右手移动 · 左手动作）",
                                           variable=self.hands_var, value="dual",
                                           command=self._param_changed)
            self.rb_dual.grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=1)
            self.rb_single = ttk.Radiobutton(box2, text="单手模式（一只手全包）",
                                             variable=self.hands_var, value="single",
                                             command=self._param_changed)
            self.rb_single.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=1)

            # 头动光标：独立于手势系统的开关
            self.head_var = tk.BooleanVar(
                value=bool(self.settings.get("head_enabled", False)))
            ttk.Checkbutton(box2, text="头动光标（头指向哪光标去哪）",
                            variable=self.head_var,
                            command=self._param_changed).grid(
                row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(4, 0))
            ttk.Label(box2, text="开启后与手势完全独立，可自由组合",
                      foreground="#8888aa").grid(
                row=4, column=0, columnspan=2, sticky="w", padx=6)

            ttk.Label(box2, text="灵敏度").grid(row=5, column=0, sticky="w", padx=6, pady=(4, 0))
            self.sens_var = tk.DoubleVar(value=float(self.settings.get("sensitivity", 1.0)))
            ttk.Scale(box2, from_=0.5, to=3.0, variable=self.sens_var,
                      command=lambda v: self._param_changed()).grid(
                row=5, column=1, sticky="we", padx=6)
            ttk.Label(box2, text="触摸灵敏度").grid(row=6, column=0, sticky="w", padx=6, pady=3)
            self.touch_var = tk.DoubleVar(
                value=float(self.settings.get("touch_sensitivity", 0.55)))
            ttk.Scale(box2, from_=0.3, to=0.9, variable=self.touch_var,
                      command=lambda v: self._param_changed()).grid(
                row=6, column=1, sticky="we", padx=6, pady=3)
            ttk.Label(box2, text="越小越需贴紧 · 越大越易触发",
                      foreground="#8888aa").grid(
                row=7, column=0, columnspan=2, sticky="w", padx=6)
            ttk.Label(box2, text="平滑度").grid(row=8, column=0, sticky="w", padx=6, pady=3)
            self.smooth_var = tk.DoubleVar(value=float(self.settings.get("smoothing", 0.35)))
            ttk.Scale(box2, from_=0.05, to=0.9, variable=self.smooth_var,
                      command=lambda v: self._param_changed()).grid(
                row=8, column=1, sticky="we", padx=6, pady=3)
            ttk.Label(box2, text="滚轮速度").grid(row=9, column=0, sticky="w", padx=6, pady=3)
            self.scroll_var = tk.DoubleVar(value=float(self.settings.get("scroll_speed", 1.0)))
            ttk.Scale(box2, from_=0.3, to=3.0, variable=self.scroll_var,
                      command=lambda v: self._param_changed()).grid(
                row=9, column=1, sticky="we", padx=6, pady=3)
            ttk.Label(box2, text="头动灵敏度").grid(row=10, column=0, sticky="w", padx=6, pady=3)
            self.head_gain_var = tk.DoubleVar(
                value=float(self.settings.get("head_sensitivity", 1.0)))
            ttk.Scale(box2, from_=0.5, to=2.5, variable=self.head_gain_var,
                      command=lambda v: self._param_changed()).grid(
                row=10, column=1, sticky="we", padx=6, pady=3)
            ttk.Label(box2, text="头部转动放大倍率（未标定时生效）",
                      foreground="#8888aa").grid(
                row=11, column=0, columnspan=2, sticky="w", padx=6)

            self.mode_var = tk.StringVar(value=self.settings.get("mode", "absolute"))
            ttk.Radiobutton(box2, text="绝对定位（手到哪光标到哪）",
                            variable=self.mode_var, value="absolute",
                            command=self._param_changed).grid(
                row=12, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            ttk.Radiobutton(box2, text="相对移动（触摸板式）",
                            variable=self.mode_var, value="relative",
                            command=self._param_changed).grid(
                row=13, column=0, columnspan=2, sticky="w", padx=6, pady=2)

            self.mirror_var = tk.BooleanVar(value=bool(self.settings.get("mirror", True)))
            ttk.Checkbutton(box2, text="镜像画面（推荐开启）", variable=self.mirror_var,
                            command=self._param_changed).grid(
                row=14, column=0, columnspan=2, sticky="w", padx=6, pady=2)
            self.lm_var = tk.BooleanVar(value=bool(self.settings.get("show_landmarks", True)))
            ttk.Checkbutton(box2, text="显示手部标记", variable=self.lm_var,
                            command=self._param_changed).grid(
                row=15, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 0))
            self.monitor_var = tk.BooleanVar(
                value=bool(self.settings.get("monitor_enabled", True)))
            ttk.Checkbutton(box2, text="右下角监视窗（悬浮置顶·可缩放）",
                            variable=self.monitor_var,
                            command=self._toggle_monitor).grid(
                row=16, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 0))
            self.game_var = tk.BooleanVar(
                value=bool(self.settings.get("game_mode", False)))
            ttk.Checkbutton(box2, text="🎮 游戏模式（战争雷霆等）",
                            variable=self.game_var,
                            command=self._param_changed).grid(
                row=17, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 0))
            ttk.Label(box2, text="SendInput 注入+4指=Tab锁定目标",
                      foreground="#8888aa").grid(
                row=18, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))
            box2.columnconfigure(1, weight=1)

            # 帮助
            box3 = ttk.Labelframe(right, text=" 手势对照表 ")
            box3.pack(fill="both", expand=True, pady=4)
            self.help_dual = (
                "双手模式：\n"
                "右手（食指尖）       移动光标\n"
                "左手仅捏合手势：\n"
                "拇指+食指触摸      单击（长按≈0.5秒=拖拽）\n"
                "拇指+中指触摸      右键单击\n"
                "拇指+无名指触摸    上滑（持续向上滚动）\n"
                "拇指+小指触摸      下滑（持续向下滚动）\n"
                "其他手指状态（1/2/3/4/5 指、握拳）无动作\n\n"
                "手离开画面          鼠标静止不误触"
            )
            self.help_single = (
                "单手模式：\n"
                "1 指（食指）        移动光标\n"
                "2 指（食+中）       左键单击\n"
                "3 指（食中无名）    右键单击\n"
                "4 指（无拇指）      滚轮（进入时手在上/下半屏=上/下滚，方向锁定）\n"
                "5 指（张开手掌）    按住左键拖拽\n"
                "拇指+食指捏合       按住左键拖拽（松开释放）\n"
                "手离开画面          鼠标静止不误触"
            )
            help_tip = ("\n\n提示：双手模式需双手同时入镜且左右分开；\n"
                        "推荐开启镜像并使用 MediaPipe 引擎。\n"
                        "「头动光标」开启后，光标由头部朝向驱动，\n"
                        "与手势系统完全独立、可自由组合；\n"
                        "首次使用建议先点「头动标定」提高指向精度。\n"
                        "🎮 游戏模式：输入改 SendInput（游戏可识别），\n"
                        "左手 4 指（无拇指）= Tab 锁定目标，\n"
                        "捏合长按 = 按住左键（持续开火），\n"
                        "建议开启头动光标+调低头动灵敏度。")
            self._help_tip = help_tip
            self.help_label = ttk.Label(box3, text=self._help_text(),
                                        justify="left", wraplength=280,
                                        foreground="#d0d0e0")
            self.help_label.pack(anchor="nw", padx=8, pady=6)

        def _help_text(self):
            base = (self.help_dual if self.hands_var.get() == "dual"
                    else self.help_single)
            return base + self._help_tip

        def _update_engine_note(self):
            note = ""
            if not mediapipe_available():
                note = "本机未检测到 mediapipe 包，将使用肤色识别引擎。\n"
                note += "如需更高精度：pip install mediapipe"
            self.engine_note.config(text=note)

        # -- 启动/停止 --------------------------------------------------------
        def _toggle_start(self):
            if self.worker and self.worker.is_alive():
                self._stop_worker()
            else:
                self._start_worker()

        def _start_worker(self):
            self._sync_settings()
            self._stop_worker()  # 清理旧线程
            self.frame_queue = queue.Queue(maxsize=2)
            self.btn_start.config(text="■  停 止", state="normal")
            self.status_bar.config(text="状态：正在打开摄像头…")
            self.worker = CameraWorker(
                dict(self.settings),
                on_frame=lambda payload: self._on_worker_frame(payload),
                on_error=lambda msg: self.after(0, self._on_worker_error, msg),
                dry_run=self.dry_run,
            )
            self.worker.start()

        def _stop_worker(self):
            if self.worker:
                self.worker.request_stop()
                self.worker.join(timeout=2.5)
                self.worker = None
            self.btn_start.config(text="▶  开 始 识 别")
            self.status_bar.config(text="状态：已停止    |    引擎：-    |    FPS：-    |    手势：-")

        def _sync_settings(self):
            self.settings["camera"] = self.cam_var.get()
            self.settings["engine"] = self.eng_var.get()
            self.settings["hands_mode"] = self.hands_var.get()
            self.settings["sensitivity"] = round(float(self.sens_var.get()), 2)
            self.settings["touch_sensitivity"] = round(float(self.touch_var.get()), 2)
            self.settings["scroll_speed"] = round(float(self.scroll_var.get()), 2)
            self.settings["head_sensitivity"] = round(float(self.head_gain_var.get()), 2)
            self.settings["head_enabled"] = bool(self.head_var.get())
            self.settings["game_mode"] = bool(self.game_var.get())
            self.settings["smoothing"] = round(float(self.smooth_var.get()), 2)
            self.settings["mode"] = self.mode_var.get()
            self.settings["mirror"] = bool(self.mirror_var.get())
            self.settings["show_landmarks"] = bool(self.lm_var.get())
            self.settings["monitor_enabled"] = bool(self.monitor_var.get())
            m = self._live_monitor()
            if m is not None:
                try:
                    self.settings["monitor_mode"] = m.mode_var.get()
                    self.settings["monitor_w"] = max(120, m.winfo_width())
                    self.settings["monitor_h"] = max(90, m.winfo_height())
                except Exception:
                    pass
            save_settings(self.settings)

        def _param_changed(self):
            self._sync_settings()
            self.help_label.config(text=self._help_text())
            if self.worker and self.worker.is_alive():
                self.worker.apply_settings(dict(self.settings))

        def _recenter_head(self):
            """把当前头部姿态设为零点（解决自然坐姿不正对导致的偏移）。"""
            if self.worker and self.worker.is_alive():
                self.settings["head_recenter"] = True
                self.worker.apply_settings({"head_recenter": True})

        # -- 头动标定 --------------------------------------------------------------
        def _start_calibration(self):
            """标定流程：确保识别运行 -> 打开全屏九点标定窗。"""
            self._sync_settings()
            if self.worker and self.worker.is_alive():
                self._open_calib_window()
            else:
                self._start_worker()
                self.after(1800, self._open_calib_window)

        def _open_calib_window(self):
            if self._closed or not (self.worker and self.worker.is_alive()):
                return
            self.worker.apply_settings({"calibrating": True})
            try:
                self._calib_win = CalibrationWindow(self, on_done=self._on_calib_done)
            except Exception as e:
                self.worker.apply_settings({"calibrating": False})
                messagebox.showerror(APP_NAME, "标定窗口创建失败：%s" % e)

        def _on_calib_done(self, coeffs, n_samples, err=None):
            if self.worker and self.worker.is_alive():
                self.worker.apply_settings({"calibrating": False})
            if self._closed:
                return
            if coeffs is not None:
                self.settings["head_calib"] = coeffs
                save_settings(self.settings)
                msg = ("标定完成！已用 %d 个采样点拟合你的个人头部朝向映射。" % n_samples)
                if err is not None:
                    msg += "\n\n平均映射误差 %.1f%%（屏幕对角线比例）。" % (err * 100.0)
                    if err > 0.08:
                        msg += "\n误差偏大，建议重新标定一次。"
                    else:
                        msg += "\n精度良好。"
                messagebox.showinfo(APP_NAME, msg)
            else:
                messagebox.showwarning(
                    APP_NAME,
                    "标定未完成（样本不足或已取消）。\n"
                    "标定时请保持正对摄像头，头部依次转向每个圆点方向并稍作停留。")

        # -- 右下角监视窗 ----------------------------------------------------------
        def _live_monitor(self):
            m = getattr(self, "monitor", None)
            if m is not None:
                try:
                    if m.winfo_exists():
                        return m
                except Exception:
                    pass
            self.monitor = None
            return None

        def _toggle_monitor(self):
            self._sync_settings()
            if self.monitor_var.get():
                self._open_monitor()
            else:
                self._close_monitor()

        def _open_monitor(self):
            m = self._live_monitor()
            if m is not None:
                m.deiconify()
                m.lift()
                return
            self.monitor = MonitorWindow(self, dict(self.settings))
            self.monitor_var.set(True)

        def _close_monitor(self):
            m = self._live_monitor()
            if m is not None:
                try:
                    m.destroy()
                except Exception:
                    pass
            self.monitor = None
            self.monitor_var.set(False)

        def _update_monitor(self, payload):
            m = self._live_monitor()
            if m is None or Image is None:
                return
            try:
                mode = m.mode_var.get()
                frame = payload["frames"].get(mode, payload["frames"]["full"])
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                w, h = m.img_label.winfo_width(), m.img_label.winfo_height()
                if w > 20 and h > 20:
                    img = img.resize((w, h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                m.img_label.configure(image=photo)
                m.img_label.image = photo
                m.set_gesture(payload["info"])
            except Exception:
                pass

        # -- 帧回传 ------------------------------------------------------------
        def _on_worker_frame(self, payload):
            # 主线程渲染；队列只保留最新一帧
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(payload)
                except queue.Empty:
                    pass

        def _poll(self):
            if self._closed:
                return
            try:
                minimized = (self.state() == "iconic")
            except Exception:
                minimized = False
            try:
                payload = self.frame_queue.get_nowait()
                self._stat.update(payload["info"])
                # 主窗口画面：最小化/后台时不渲染（省 CPU 给识别线程）
                if not minimized and Image is not None:
                    img = Image.fromarray(
                        cv2.cvtColor(payload["frames"]["full"], cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(img)
                    self.video_label.configure(image=photo, text="")
                    self.video_label.image = photo
                # 右下角监视窗始终更新（最小化主窗口后仍可见）
                self._update_monitor(payload)
            except queue.Empty:
                pass
            except Exception as e:  # noqa: BLE001
                print("渲染错误:", e)
            self._refresh_status()
            self.after(20, self._poll)

        def _refresh_status(self):
            s = self._stat
            g = GESTURE_LABEL_CN.get(s["gesture"], "-")
            hand = "有手" if s["hand"] else "无手"
            running = bool(self.worker and self.worker.is_alive())
            state = "运行中" if running else "未启动"
            hands_txt = ""
            if s.get("hands_mode") == "dual":
                hands_txt = "    |    左手:%s 右手:%s" % (
                    "✓" if s.get("left") else "—",
                    "✓" if s.get("right") else "—")
            if s.get("head_on"):
                face = "✓" if s.get("face") else "—"
                hands_txt += "    |    头动·人脸:%s" % face
            text = ("状态：%s    |    引擎：%s    |    FPS：%.0f    |    "
                    "手势：%s（%s）%s" % (state, s["engine"], s["fps"], g, hand,
                                          hands_txt))
            # 只在内容变化时更新控件，避免每帧无谓的字符串重绘
            if text != getattr(self, "_last_status_text", None):
                self.status_bar.config(text=text)
                self._last_status_text = text

        def _on_worker_error(self, msg):
            self._stop_worker()
            if self._closed:
                return
            try:
                with open(os.path.join(BASE_DIR, "worker_error.txt"),
                          "w", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            if not self.dry_run:
                messagebox.showerror(APP_NAME, msg)

        # -- 退出 ----------------------------------------------------------------
        def _on_close(self):
            self._sync_settings()
            self._stop_worker()
            self._closed = True
            try:
                self._close_monitor()
            except Exception:
                pass
            self.destroy()

    return App


# ---------------------------------------------------------------------------
# 进程优先级
# ---------------------------------------------------------------------------
def boost_process_priority():
    """Windows 下把进程优先级提到 ABOVE_NORMAL。

    手势控制时程序窗口必然在后台（用户在别的应用里操作），
    Windows 会压低后台进程的调度，导致识别帧率掉、光标移动卡顿。
    提到 ABOVE_NORMAL（略高于普通，不影响系统）保证后台也流畅。
    """
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.windll.kernel32
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00008000)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 无界面自检
# ---------------------------------------------------------------------------
def run_headless_test(engine_kind="auto", camera_choice="auto", seconds=6):
    print("=" * 60)
    print("手势鼠标 无界面自检  v%s" % VERSION)
    print("Python %s" % sys.version.split()[0])
    print("OpenCV %s  NumPy %s" % (cv2.__version__, np.__version__))
    print("mediapipe: %s" % ("可用" if mediapipe_available() else "未安装（使用肤色引擎）"))
    print("=" * 60)

    settings = dict(DEFAULT_SETTINGS)
    settings["engine"] = engine_kind
    settings["camera"] = camera_choice

    try:
        engine, note = make_engine(engine_kind)
        print("引擎:", note)
    except Exception as e:
        print("引擎初始化失败:", e)
        return 1

    cap = _open_camera(camera_choice)
    if cap is None:
        print("!! 未找到可用摄像头")
        print("   请检查：摄像头是否被其他程序占用 / 是否被系统禁用")
        return 1
    print("摄像头已打开: %.0fx%.0f" % (cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                       cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    mouse = DryRunMouse()
    interp = GestureInterpreter(mouse, settings)
    frames = hands = 0
    gestures_seen = {}
    t0 = time.monotonic()
    last_t = t0
    try:
        while time.monotonic() - t0 < seconds:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.02)
                continue
            now = time.monotonic()
            dt = min(now - last_t, 0.2)
            last_t = now
            if settings["mirror"]:
                frame = cv2.flip(frame, 1)
            if engine.name == "MediaPipe":
                res = engine.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                res = engine.detect(frame)
            eff = interp.update(res, dt)
            frames += 1
            if res.hand_found:
                hands += 1
            gestures_seen[eff] = gestures_seen.get(eff, 0) + 1
            if hands > 0 and hands % 30 == 1:
                print("  [检测到手] 伸出指数: %d 手势: %s" % (
                    res.count, GESTURE_LABEL_CN.get(eff, eff)))
    finally:
        cap.release()
        engine.close()

    print("-" * 60)
    print("处理帧数: %d   检测到手: %d (%.0f%%)" % (
        frames, hands, 100.0 * hands / max(frames, 1)))
    print("手势统计:", {GESTURE_LABEL_CN.get(k, k): v
                        for k, v in gestures_seen.items()})
    print("模拟鼠标动作:", mouse.events)
    ok = frames > 5
    print("-" * 60)
    if ok:
        print("自检通过：摄像头与识别管线工作正常。")
        if hands == 0:
            print("注意：测试期间未检测到手。请把手伸到摄像头前重试；")
            print("      肤色引擎需要良好光线与干净背景。")
    else:
        print("自检失败：摄像头未产出有效帧。")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    # Windows 控制台输出中文修复（GBK 代码页 -> UTF-8）
    try:
        if sys.platform == "win32" and sys.stdout is not None:
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    boost_process_priority()   # 后台运行时保持流畅
    if "--headless-test" in sys.argv:
        sys.exit(run_headless_test())
    if "--smoke" in sys.argv:
        # 冒烟测试：结果写文件（windowed exe 无控制台），并硬退出兜底
        smoke_log = os.path.join(BASE_DIR, "smoke_result.txt")
        try:
            if os.path.exists(smoke_log):
                os.remove(smoke_log)
        except Exception:
            pass
        App = build_gui(dry_run=True)   # 冒烟测试不碰真实鼠标
        try:
            app = App()
        except Exception as e:
            try:
                with open(smoke_log, "w", encoding="utf-8") as f:
                    f.write("GUI_CREATE_FAIL: %s" % e)
            except Exception:
                pass
            print("GUI 创建失败（可能当前会话无桌面权限）:", e)
            sys.exit(1)
        smoke = {"ok": False}

        def _log(s):
            try:
                with open(smoke_log, "a", encoding="utf-8") as f:
                    f.write("\n" + s)
            except Exception:
                pass

        def auto_close():
            smoke["ok"] = True
            _log("AUTO_CLOSE_OK")
            try:
                _log("step:sync_settings")
                app._sync_settings()
                _log("step:stop_worker")
                app._stop_worker()
                _log("step:close_monitor")
                app._close_monitor()
                app._closed = True
                _log("step:destroy")
                app.destroy()
                _log("step:destroyed")
            except Exception as e:
                _log("ON_CLOSE_ERR: %s" % e)
            _log("step:os_exit")
            try:
                os._exit(0)   # 硬退出，保证冒烟测试必结束
            except Exception:
                pass

        app.after(1000, app._toggle_start)   # 自动启动
        app.after(4500, auto_close)
        app.mainloop()
        print("GUI 冒烟测试完成: %s" % ("通过" if smoke["ok"] else "未完成"))
        sys.exit(0 if smoke["ok"] else 1)

    App = build_gui()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
