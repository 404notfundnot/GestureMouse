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

# 阶段 B：左手 1 指 -> 左键单击一次（保留的原手势）
left1 = make_hand("left", [False, True, False, False, False], pointer=(0.3, 0.4))
step(interp, left1, right, n=12)
print("阶段B(左手1指) events:", dict(mouse.events))
assert mouse.events["lclick"] == 1

# 阶段 C：左手 2 指 -> 右键单击（保留的原手势）
cool(); step(interp, right, n=6)
left2 = make_hand("left", [False, True, True, False, False])
step(interp, left2, right, n=12)
print("阶段C(左手2指) events:", dict(mouse.events))
assert mouse.events["rclick"] == 1

# 阶段 D：左手 3 指 -> 双击（保留的原手势）
cool(); step(interp, right, n=6)
left3 = make_hand("left", [False, True, True, True, False])
step(interp, left3, right, n=12)
print("阶段D(左手3指) events:", dict(mouse.events))
assert mouse.events["dclick"] == 1

# 阶段 E：左手 5 指张开 -> 无动作（拖拽改用拇+食指长按，阶段 I 覆盖）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
left5 = make_hand("left", [True, True, True, True, True])
step(interp, left5, right, n=15)
d = delta(mouse, before)
print("阶段E(左手5指张开=无动作) 增量:", d)
assert d["down"] == 0 and d["up"] == 0 and d["lclick"] == 0 \
    and d["rclick"] == 0 and d["dclick"] == 0 and d["scroll"] == 0

# 阶段 F：左手 4 指 -> 滚轮（方向锁定：进入瞬间所在半屏决定方向）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
left4 = make_hand("left", [False, True, True, True, True], pointer=(0.3, 0.2))
step(interp, left4, right, n=12)      # 上半屏进入 -> 锁定上滚
d1 = delta(mouse, before)
print("阶段F(4指上半屏进入=上滚) 增量:", d1)
assert d1["scroll"] >= 1 and d1["scroll_sum"] > 0
# 关键：期间手移到下半屏，方向必须不变（修复"滚轮乱动"）
before = dict(mouse.events)
left4b = make_hand("left", [False, True, True, True, True], pointer=(0.3, 0.85))
step(interp, left4b, right, n=12)
d2 = delta(mouse, before)
print("阶段F(期间手移到下半屏,方向应保持不变) 增量:", d2)
assert d2["scroll_sum"] > 0           # 仍向上滚，不随手位置翻转
# 收起手势后在下半屏重新进入 -> 锁定下滚
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, left4b, right, n=12)
d3 = delta(mouse, before)
print("阶段F(下半屏重新进入=下滚) 增量:", d3)
assert d3["scroll_sum"] < 0

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
step(interp, left4b, right, n=24)     # 越过冷静期+滚轮确认期，4 指正常生效（下半屏=下滚）
d3 = delta(mouse, before)
print("阶段O(冷静期过后4指恢复生效) 增量:", d3)
assert d3["scroll_sum"] < 0

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

# 阶段 Q：连续两次 3 指双击（间隔约 0.9s）-> 两次双击（独立冷却 0.4s 生效）
cool(); step(interp, right, n=8)
before = dict(mouse.events)
step(interp, left3, right, n=12)   # 第一次双击
step(interp, right, n=8)
step(interp, left3, right, n=12)   # 冷却后第二次双击
d = delta(mouse, before)
print("阶段Q(连续两次双击) 增量:", d)
assert d["dclick"] == 2

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

# 阶段 N：方向锁定用原始位置（EMA 平滑惯性不应导致方向锁反）
mouse4 = gm.DryRunMouse()
interp4 = gm.GestureInterpreter(mouse4, dict(gm.DEFAULT_SETTINGS, hands_mode="dual",
                                             mode="absolute", smoothing=0.05))
right4 = make_hand("right", [False, True, False, False, False], pointer=(0.6, 0.5))
hand_low = make_hand("left", [False] * 5, pointer=(0.3, 0.8))
hand_high = make_hand("left", [False, True, True, True, True], pointer=(0.3, 0.2))
step(interp4, right4, hand_low, n=6)        # 左手先停在下半屏（平滑值仍偏下）
before = dict(mouse4.events)
step(interp4, right4, hand_high, n=12)      # 4 指滚轮，手瞬移到上半屏
d = delta(mouse4, before)
print("阶段N(平滑惯性下方向锁定应=上滚) 增量:", d)
assert d["scroll_sum"] > 0

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

# 阶段 S：滚轮速度设置生效（scroll_speed=2.0 约为 1.0 的两倍）
mouse5 = gm.DryRunMouse()
interp5 = gm.GestureInterpreter(mouse5, dict(gm.DEFAULT_SETTINGS, hands_mode="dual",
                                             scroll_speed=2.0, mode="absolute"))
right5 = make_hand("right", [False, True, False, False, False], pointer=(0.6, 0.5))
step(interp5, right5, left4, n=8)     # left4: 4 指上半屏（阶段 F 定义）
before = dict(mouse5.events)
step(interp5, right5, left4, n=15)
d = delta(mouse5, before)
print("阶段S(滚轮速度2x) 增量:", d)
assert d["scroll_sum"] >= 6           # 1x 时 15 帧约 3 格，2x 应约 6 格

engine.close()
skin.close()
print("ALL OK")
