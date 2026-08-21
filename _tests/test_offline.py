# -*- coding: utf-8 -*-
"""离线验证：照片识别 + 单手/双手手势状态机 + 模拟鼠标动作（可删除）"""
import sys, os, time
try:
    import ctypes
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import gesture_mouse as gm


def make_hand(label, fingers, pointer=(0.5, 0.4), pinch=False, thumb_touch=0):
    h = gm.HandInfo()
    h.found = True
    h.label = label
    h.fingers = fingers
    h.pointer = pointer
    h.pinch = pinch
    h.thumb_touch = thumb_touch
    return h


def make_result(*hands):
    r = gm.GestureResult()
    r.hands = list(hands)
    return r


def step(interp, *hands, n=1):
    """以真实 30fps 帧间隔推帧（防抖按真实时间计）。"""
    for _ in range(n):
        interp.update(make_result(*hands), dt=1/30)
        time.sleep(1/30)


def step_res(interp, res, n=1, allow_move=True):
    """直接推一个已构造好的 GestureResult（眼动模式用）。"""
    for _ in range(n):
        interp.update(res, dt=1/30, allow_move=allow_move)
        time.sleep(1/30)


def cool():
    """模拟用户两次动作之间的自然间隔（超过点击冷却 0.7s）。"""
    time.sleep(0.9)


def delta(mouse, before):
    return {k: mouse.events[k] - before[k] for k in before}


# ---------- 1. MediaPipe 双手识别：woman_hands.jpg 是双手祈祷图 ----------
img = cv2.imread("_tests/test_hand.jpg")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
engine, note = gm.make_engine("mediapipe")
print("engine:", note)
res = engine.detect(rgb, mirror=True)
print("检测到手数量:", len(res.hands))
for h in res.hands:
    print("  手: label=%s fingers=%s count=%d pinch=%s thumb_touch=%d" %
          (h.label, h.fingers, h.count, h.pinch, h.thumb_touch))
print("  用户左手存在:", res.hand("left") is not None,
      " 用户右手存在:", res.hand("right") is not None)

# ---------- 2. 双手模式：右手移动 + 左手动作 ----------
settings = dict(gm.DEFAULT_SETTINGS, hands_mode="dual", mode="absolute", smoothing=0.5)

mouse = gm.DryRunMouse()
interp = gm.GestureInterpreter(mouse, settings)
right = make_hand("right", [False, True, False, False, False], pointer=(0.6, 0.4))
# 阶段 A：仅右手（应持续移动，无点击）
step(interp, right, n=10)
print("阶段A(仅右手) events:", dict(mouse.events))
assert mouse.events["move"] > 5 and mouse.events["lclick"] == 0

# 阶段 B：左手 1 指 -> 无动作（手指数手势已删除，仅保留捏合判定）
left1 = make_hand("left", [False, True, False, False, False], pointer=(0.3, 0.4))
before = dict(mouse.events)
step(interp, left1, right, n=12)
d = delta(mouse, before)
print("阶段B(左手1指=无动作) 增量:", d)
assert d["lclick"] == 0 and d["rclick"] == 0 and d["dclick"] == 0 \
    and d["scroll"] == 0 and d["down"] == 0

# 阶段 C：左手 2 指 -> 无动作
cool(); step(interp, right, n=6)
left2 = make_hand("left", [False, True, True, False, False])
before = dict(mouse.events)
step(interp, left2, right, n=12)
d = delta(mouse, before)
print("阶段C(左手2指=无动作) 增量:", d)
assert d["lclick"] == 0 and d["rclick"] == 0 and d["scroll"] == 0

# 阶段 D：左手 3 指 -> 无动作
cool(); step(interp, right, n=6)
left3 = make_hand("left", [False, True, True, True, False])
before = dict(mouse.events)
step(interp, left3, right, n=12)
d = delta(mouse, before)
print("阶段D(左手3指=无动作) 增量:", d)
assert d["dclick"] == 0 and d["lclick"] == 0 and d["scroll"] == 0

# 阶段 E：左手 5 指张开 -> 无动作
cool(); step(interp, right, n=8)
before = dict(mouse.events)
left5 = make_hand("left", [True, True, True, True, True])
step(interp, left5, right, n=15)
d = delta(mouse, before)
print("阶段E(左手5指张开=无动作) 增量:", d)
assert d["down"] == 0 and d["up"] == 0 and d["lclick"] == 0 \
    and d["rclick"] == 0 and d["dclick"] == 0 and d["scroll"] == 0

# 阶段 F：左手 4 指 -> 无动作（滚轮已由拇+无名指/小指承担）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
left4 = make_hand("left", [False, True, True, True, True], pointer=(0.3, 0.2))
step(interp, left4, right, n=12)
d1 = delta(mouse, before)
print("阶段F(4指上半屏=无动作) 增量:", d1)
assert d1["scroll"] == 0 and d1["scroll_sum"] == 0
before = dict(mouse.events)
left4b = make_hand("left", [False, True, True, True, True], pointer=(0.3, 0.85))
step(interp, left4b, right, n=12)
d2 = delta(mouse, before)
print("阶段F(4指下半屏=无动作) 增量:", d2)
assert d2["scroll"] == 0 and d2["scroll_sum"] == 0

# ---------- 3. 新增：拇指触摸系列 ----------
# 阶段 H：拇指+食指 短触摸（0.2s）-> 单击
#（真实捏合姿态：中指/无名指/小指伸直）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
pinch1 = make_hand("left", [False, False, True, True, True], thumb_touch=1)
step(interp, pinch1, right, n=6)      # 0.2s 捏合
step(interp, right, n=8)              # 松开
d = delta(mouse, before)
print("阶段H(拇+食短触摸=单击) 增量:", d)
assert d["lclick"] == 1 and d["down"] == 0 and d["up"] == 0

# 阶段 I：拇指+食指 长按（0.7s）-> 拖拽，松开释放
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, pinch1, right, n=22)     # 0.73s 长按
d1 = delta(mouse, before)
print("阶段I(拇+食长按=拖拽) 按住期间增量:", d1)
assert d1["down"] == 1 and d1["lclick"] == 0
before = dict(mouse.events)
step(interp, right, n=8)              # 松开
d2 = delta(mouse, before)
print("阶段I后(松开释放) 增量:", d2)
assert d2["up"] == 1

# 阶段 J：拇指+中指触摸 -> 右键（真实姿态：无名/小指伸直）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
pinch2 = make_hand("left", [False, False, False, True, True], thumb_touch=2)
step(interp, pinch2, right, n=12)
d = delta(mouse, before)
print("阶段J(拇+中=右键) 增量:", d)
assert d["rclick"] == 1

# 阶段 K：拇指+无名指触摸 -> 上滑（持续向上滚动，小指伸直）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
pinch3 = make_hand("left", [False, False, False, False, True], thumb_touch=3)
step(interp, pinch3, right, n=18)
d = delta(mouse, before)
print("阶段K(拇+无名指=上滑) 增量:", d)
assert d["scroll"] >= 1 and d["scroll_sum"] > 0

# 阶段 L：拇指+小指触摸 -> 下滑（持续向下滚动，食中无名伸直）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
pinch4 = make_hand("left", [False, True, True, True, False], thumb_touch=4)
step(interp, pinch4, right, n=18)
d = delta(mouse, before)
print("阶段L(拇+小指=下滑) 增量:", d)
assert d["scroll"] >= 1 and d["scroll_sum"] < 0

# 阶段 O：下滑松开后手指自然变 4 指（手在下半屏）——冷静期内不应继续滚
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, pinch4, right, n=12)     # 拇+小指 -> 下滑生效
d1 = delta(mouse, before)
print("阶段O(拇+小指下滑) 增量:", d1)
assert d1["scroll_sum"] < 0
before = dict(mouse.events)
step(interp, left4b, right, n=8)      # 松开，手指变 4 指（冷静期 0.45s 内）
d2 = delta(mouse, before)
print("阶段O(松开变4指,冷静期内应基本不滚) 增量:", d2)
assert d2["scroll"] <= 2              # 仅防抖窗口惯性，不得持续滚动
before = dict(mouse.events)
step(interp, left4b, right, n=24)     # 4 指持续保持：始终无动作（手指数手势已删除）
d3 = delta(mouse, before)
print("阶段O(松开后4指始终无动作) 增量:", d3)
assert d3["scroll_sum"] == 0

# 阶段 G：双手都不在 -> 一切静止（先稳定回无动作态再测）
step(interp, right, n=8)
before = dict(mouse.events)
step(interp, n=8)
print("阶段G(无手) events 增量:", delta(mouse, before))
assert all(v == 0 for v in delta(mouse, before).values())

# 阶段 P：捏合快速两连（间隔约 0.2s）-> 两次单击（系统层合成双击）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, pinch1, right, n=6)   # 第一次捏合 0.2s
step(interp, right, n=3)           # 松开（触发第一次单击）
step(interp, pinch1, right, n=6)   # 快速再捏
step(interp, right, n=6)           # 再松开（触发第二次单击）
d = delta(mouse, before)
print("阶段P(快速两连捏=两次单击) 增量:", d)
assert d["lclick"] == 2            # 两次单击间隔 <0.5s，系统识别为双击

# 阶段 Q 已删除：双手模式左手 3 指双击手势随"手指数判定"一起删除，
# 双击现在通过"快速两连捏"（阶段 P）由系统合成。

# 阶段 M：捏合释放防抖（识别闪断 1 帧不应误触发单击/拖拽）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, pinch1, right, n=6)      # 捏合 0.2s
step(interp, right, n=1)              # 闪断 1 帧
step(interp, pinch1, right, n=2)      # 继续捏
step(interp, right, n=6)              # 真正松开 -> 单击一次
d = delta(mouse, before)
print("阶段M(捏合闪断1帧不误触发) 增量:", d)
assert d["lclick"] == 1 and d["down"] == 0

# 阶段 N 已删除：4 指滚轮方向锁定随"手指数手势"一起删除，
# 滚轮现在只有拇+无名指（上滑）/拇+小指（下滑）。

# ---------- 4. 单手模式回归（不变） ----------
mouse2 = gm.DryRunMouse()
interp2 = gm.GestureInterpreter(mouse2, dict(gm.DEFAULT_SETTINGS, hands_mode="single"))
seq = [[False, True, False, False, False]] * 10 + \
      [[False, True, True, False, False]] * 10 + \
      [[False, True, False, False, False]] * 5
for f in seq:
    step(interp2, make_hand("right", f))
print("单手回归 events:", dict(mouse2.events))
assert mouse2.events["lclick"] == 1

# ---------- 5. 肤色引擎冒烟 ----------
skin, _ = gm.make_engine("skin")
r3 = skin.detect(cv2.imread("_tests/test_hand.jpg"), mirror=True)
print("肤色引擎: 检测手数=%d" % len(r3.hands))

# ---------- 6. 真实鼠标控制器冒烟（wintypes 导入修复回归，移动 3px 无感） ----------
mc = gm.MouseController()
mc.move_rel(3, 4)
print("MouseController.move_rel OK, 当前光标:", mc.position())

# ---------- 7. 相对移动模式（触摸板式）解释器测试 ----------
mouse3 = gm.DryRunMouse()
interp3 = gm.GestureInterpreter(mouse3, dict(gm.DEFAULT_SETTINGS,
                                             hands_mode="dual", mode="relative",
                                             smoothing=0.5))
right_r = make_hand("right", [False, True, False, False, False], pointer=(0.75, 0.5))
step(interp3, right_r, n=10)
print("相对移动模式 events:", dict(mouse3.events))
assert mouse3.events["move"] > 3

# ---------- 8. 旋转鲁棒性：合成 landmark 测试 ----------
class LM:
    __slots__ = ("x", "y", "z")
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z


def mk_lm():
    """张开手掌的关键点：手方向向上（手腕->中指掌根）。"""
    lm = [LM(0.5, 0.5, 0.0) for _ in range(21)]
    lm[0] = LM(0.5, 0.9, 0.0)     # 手腕
    lm[9] = LM(0.5, 0.75, 0.0)    # 中指掌根
    lm[5] = LM(0.44, 0.76, 0.0)   # 食指掌根
    lm[17] = LM(0.56, 0.78, 0.0)  # 小指掌根
    lm[6] = LM(0.43, 0.60, 0.0); lm[8] = LM(0.42, 0.42, 0.0)
    lm[10] = LM(0.50, 0.58, 0.0); lm[12] = LM(0.50, 0.40, 0.0)
    lm[14] = LM(0.56, 0.60, 0.0); lm[16] = LM(0.58, 0.42, 0.0)
    lm[18] = LM(0.61, 0.62, 0.0); lm[20] = LM(0.64, 0.45, 0.0)
    lm[4] = LM(0.30, 0.80, 0.0)   # 拇指尖（张开，远离小指掌根）
    lm[3] = LM(0.36, 0.78, 0.0)
    lm[2] = LM(0.38, 0.80, 0.0)
    return lm


def rot_m90(p, cx=0.5, cy=0.75):
    """绕 (cx,cy) 逆时针旋转 90 度（手方向从"向上"变为"向左"）。"""
    return LM(cx + (p.y - cy), cy - (p.x - cx), p.z)


# 8a. 正常张开手：五指都应判伸出
up = gm.MediaPipeEngine._fingers_up(mk_lm())
print("8a 张开手 fingers:", up)
assert up == [True, True, True, True, True]

# 8b. 手腕旋转 90°（手指横指）：旧 y 轴判据会误判，新判据应仍判伸出
up2 = gm.MediaPipeEngine._fingers_up([rot_m90(p) for p in mk_lm()])
print("8b 旋转90°横指 fingers:", up2)
assert up2[1] and up2[2] and up2[3] and up2[4]

# 8c. 旋转视角：拇指与食指 2D 投影重叠但深度(z)分离 -> 不得判为触摸
lm3 = mk_lm()
lm3[4] = LM(0.43, 0.80, 0.0)
lm3[8] = LM(0.42, 0.80, 0.35)    # 画面上几乎重合，但深处 0.35
t3 = gm.MediaPipeEngine._thumb_touch(lm3, 0.55)
print("8c 2D重叠但z分离 thumb_touch:", t3, "(应为 0)")
assert t3 == 0

# 8d. 真实捏合（2D 与深度都接近）-> 应判食指触摸
lm4 = mk_lm()
lm4[4] = LM(0.43, 0.80, 0.0)     # 拇指尖移到食指尖旁
lm4[8] = LM(0.42, 0.80, 0.02)
t4 = gm.MediaPipeEngine._thumb_touch(lm4, 0.55)
print("8d 真实捏合 thumb_touch:", t4, "(应为 1)")
assert t4 == 1

# 8e. 张开手掌 + 小指外展：应判 5 指全伸出
#     （旧"沿手方向投影"判据会把外展小指误判为弯曲 -> 5 指变 4 指 -> 误触滚轮）
lm5 = mk_lm()
lm5[20] = LM(0.82, 0.66, 0.0)    # 小指明显向外撇开
up5 = gm.MediaPipeEngine._fingers_up(lm5)
print("8e 张开手小指外展 fingers:", up5)
assert up5 == [True, True, True, True, True]
h5 = gm.HandInfo()
h5.found = True
h5.fingers = up5
g5, pinch5 = gm.GestureInterpreter.classify_left(h5)
print("8e 分类:", g5, "(应为 none，不是 scroll/drag)")
assert g5 == gm.GESTURE_NONE and not pinch5

# 8f. 握拳（指尖卷向手心）-> 四指都不判伸出
lm6 = mk_lm()
for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
    lm6[tip] = LM(lm6[pip].x, lm6[pip].y + 0.08, 0.0)  # 指尖卷到关节下方
up6 = gm.MediaPipeEngine._fingers_up(lm6)
print("8f 握拳 fingers:", up6)
assert up6[1:] == [False, False, False, False]

# 阶段 R：握拳（全指弯曲 + 拇指压食指）-> 无动作，不触发单击
cool(); step(interp, right, n=8)
before = dict(mouse.events)
fist = make_hand("left", [False] * 5, thumb_touch=1)
step(interp, fist, right, n=10)   # 握拳保持
step(interp, right, n=8)          # 松开拳头
d = delta(mouse, before)
print("阶段R(握拳=无动作) 增量:", d)
assert d["lclick"] == 0 and d["down"] == 0 and d["up"] == 0 \
    and d["scroll"] == 0

# 阶段 S：滚轮速度设置生效（scroll_speed=2.0 约为 1.0 的两倍，用拇+无名指上滑验证）
mouse5 = gm.DryRunMouse()
interp5 = gm.GestureInterpreter(mouse5, dict(gm.DEFAULT_SETTINGS, hands_mode="dual",
                                             scroll_speed=2.0, mode="absolute"))
right5 = make_hand("right", [False, True, False, False, False], pointer=(0.6, 0.5))
step(interp5, right5, pinch3, n=8)     # pinch3: 拇+无名指（阶段 K 定义）
before = dict(mouse5.events)
step(interp5, right5, pinch3, n=15)
d = delta(mouse5, before)
print("阶段S(滚轮速度2x,拇+无名指上滑) 增量:", d)
assert d["scroll_sum"] >= 6           # 1x 时 15 帧约 6 格，2x 应约 12 格

# ---------- 9. 头动模式 ----------
# 9a. HeadPoseEngine 真实人脸照片检测（矩阵角度版）
ge = gm.HeadPoseEngine()
img_f = cv2.imread("_tests/test_face.jpg")
gp, ginfo = ge.detect(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB), mirror=True)
print("9a 人脸检测 face=%s pose(rad)=%s" %
      (ginfo["face"], [round(v, 3) for v in gp] if gp else None))
assert ginfo["face"] and gp is not None
# 正脸照片：yaw/pitch 应接近 0 弧度
assert abs(gp[0]) < 0.15 and abs(gp[1]) < 0.2
ge.close()

# 9b. 矩阵角度提取轴向验证（合成旋转矩阵）
# 点头=绕水平轴 Rx，转头=绕竖直轴 Rz
def Rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


M9 = np.eye(4)
M9[:3, :3] = Rz(0.3) @ Rx(0.2)     # 转头 0.3 rad + 点头 0.2 rad
yaw9, pitch9 = gm.HeadPoseEngine._angles_from_matrix(M9)
print("9b 合成矩阵 yaw=%.3f(期望0.3) pitch=%.3f(期望0.2)" % (yaw9, pitch9))
assert abs(yaw9 - 0.3) < 0.01 and abs(pitch9 - 0.2) < 0.01
# 纯点头：yaw 应为 0
M9b = np.eye(4)
M9b[:3, :3] = Rx(0.25)
yaw9b, pitch9b = gm.HeadPoseEngine._angles_from_matrix(M9b)
print("9b 纯点头 yaw=%.3f(期望0) pitch=%.3f(期望0.25)" % (yaw9b, pitch9b))
assert abs(yaw9b) < 0.01 and abs(pitch9b - 0.25) < 0.01


# ---------- 10. 头动独立逻辑（HeadController 与手势解耦） ----------

class FakeHeadEngine:
    """模拟 HeadPoseEngine：按序列返回 (pose_ratio, info)。"""
    def __init__(self, seq):
        self.seq = seq
        self.i = 0
    def detect(self, rgb, mirror=True):
        r = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return r
    def close(self):
        pass


# 10a. HeadController：姿态有效 -> 独立移动鼠标（中值+EMA 平滑）
mouse_g = gm.DryRunMouse()
hc = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.5,
                            head_sensitivity=1.0), mouse_g)
hc.engine = FakeHeadEngine([((0.4, 0.1), {"face": True, "face_pts": []})] * 30)
info = {}
for _ in range(15):
    info = hc.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10a 头动移动: move=%d valid=%s" % (mouse_g.events["move"], info["valid"]))
assert mouse_g.events["move"] >= 5 and info["valid"]

# 10b. 无人脸 -> 冻结，不移动
gc2 = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.5), gm.DryRunMouse())
gc2.engine = FakeHeadEngine([(None, {"face": False, "face_pts": []})] * 10)
for _ in range(10):
    gc2.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10b 无人脸冻结 move=%d" % gc2.mouse.events["move"])
assert gc2.mouse.events["move"] == 0

# 10c. 标定映射（个人化系数优先于线性增益，角度域）
calib_pts = []
for sx, sy in [(0.5, 0.5), (0.12, 0.12), (0.88, 0.12), (0.88, 0.88),
               (0.12, 0.88), (0.5, 0.12), (0.88, 0.5), (0.5, 0.88), (0.12, 0.5)]:
    # 角度与屏幕坐标近似线性：yaw = (sx-0.5)*0.8, pitch = (sy-0.5)*0.6
    calib_pts.append(((sx - 0.5) * 0.8, (sy - 0.5) * 0.6, sx, sy))
calib_coeffs = gm.PoseCalibrator.fit(calib_pts)
assert calib_coeffs is not None
mouse_g3 = gm.DryRunMouse()
hc3 = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.5,
                             head_calib=calib_coeffs), mouse_g3)
hc3.engine = FakeHeadEngine([(((0.88 - 0.5) * 0.8, (0.12 - 0.5) * 0.6),
                              {"face": True, "face_pts": []})] * 20)
for _ in range(12):
    hc3.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10c 标定映射 move=%d" % mouse_g3.events["move"])
assert mouse_g3.events["move"] >= 5

# 10d. 解释器解耦：allow_move=False 时手势只做动作、不抢光标移动
mouse6 = gm.DryRunMouse()
interp6 = gm.GestureInterpreter(mouse6, dict(gm.DEFAULT_SETTINGS,
                                             hands_mode="dual", smoothing=0.5))
right6 = make_hand("right", [False, True, False, False, False], pointer=(0.7, 0.5))
step_res(interp6, make_result(right6), n=10, allow_move=False)
print("10d allow_move=False(右手在场但不动光标) events:", dict(mouse6.events))
assert mouse6.events["move"] == 0
# 左手捏合手势照常（手指数手势已删除，用拇+食指短触摸验证）
cool(); before = dict(mouse6.events)
left_t = make_hand("left", [False, False, True, True, True], thumb_touch=1)
step_res(interp6, make_result(left_t, right6), n=8, allow_move=False)
step_res(interp6, make_result(right6), n=8, allow_move=False)   # 松开 -> 单击
d = delta(mouse6, before)
print("10d 捏合单击(allow_move=False) 增量:", d)
assert d["lclick"] == 1 and d["move"] == 0

# 10e. 轻微持续转动：光标应持续响应（旧"增量死区"逻辑会完全卡住）
mouse_h = gm.DryRunMouse()
hc4 = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.4,
                             head_sensitivity=1.0), mouse_h)
seq = []
cur = 0.0
for _ in range(40):
    cur += 0.003              # 每帧缓慢右转 0.003 rad（约 0.17°）
    seq.append(((cur, 0.0), {"face": True, "face_pts": []}))
hc4.engine = FakeHeadEngine(seq)
for _ in range(40):
    hc4.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10e 轻微持续转动 move=%d" % mouse_h.events["move"])
assert mouse_h.events["move"] >= 15   # 旧逻辑下为 0

# 10f. 完全静止：不应漂移（死区抑制微抖）
mouse_i = gm.DryRunMouse()
hc5 = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.4), mouse_i)
hc5.engine = FakeHeadEngine([((0.0, 0.0), {"face": True, "face_pts": []})] * 15)
for _ in range(15):
    hc5.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10f 静止不漂移 move=%d" % mouse_i.events["move"])
assert mouse_i.events["move"] <= 2   # 初始 0.5 中心到目标 0.5 的偏差为 0，不移动

# 10g. 重置中心：自然坐姿不正对（姿态 0.1 rad）时一键归零
mouse_j = gm.DryRunMouse()
hc6 = gm.HeadController(dict(gm.DEFAULT_SETTINGS, smoothing=0.4,
                             head_sensitivity=1.0), mouse_j)
hc6.engine = FakeHeadEngine([((0.1, 0.04), {"face": True, "face_pts": []})] * 60)
for _ in range(12):
    hc6.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10g 重置前 last_pos=%.3f,%.3f" % mouse_j.last_pos)
assert mouse_j.last_pos[0] > 0.52          # 头微侧 -> 光标偏右
hc6.recenter()                              # 当前姿态设为零点
for _ in range(12):
    hc6.process(np.zeros((480, 640, 3), dtype=np.uint8))
print("10g 重置后 last_pos=%.3f,%.3f" % mouse_j.last_pos)
assert abs(mouse_j.last_pos[0] - 0.5) < 0.02 and abs(mouse_j.last_pos[1] - 0.5) < 0.02

# 阶段 V：游戏模式——左手 4 指（无拇指）= Tab 锁定目标
mouse8 = gm.DryRunMouse()
interp8 = gm.GestureInterpreter(mouse8, dict(gm.DEFAULT_SETTINGS, hands_mode="dual",
                                             game_mode=True))
right8 = make_hand("right", [False, True, False, False, False])
hand4 = make_hand("left", [False, True, True, True, True])
step(interp8, right8, hand4, n=8)      # 防抖 0.1s 后进入
step(interp8, right8, hand4, n=10)
print("阶段V(游戏模式4指=Tab) events:", dict(mouse8.events))
assert mouse8.events["key"] == 1
# 非游戏模式：同样手势无动作
mouse9 = gm.DryRunMouse()
interp9 = gm.GestureInterpreter(mouse9, dict(gm.DEFAULT_SETTINGS, hands_mode="dual"))
step(interp9, right8, hand4, n=12)
print("阶段V(非游戏模式4指=无动作) key=%d" % mouse9.events["key"])
assert mouse9.events["key"] == 0

engine.close()
skin.close()
print("ALL OK")
