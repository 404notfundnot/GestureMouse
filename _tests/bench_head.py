# -*- coding: utf-8 -*-
"""头动管线性能基准（真实摄像头）：人脸全帧 + 手部降频 对比 串行全帧"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import gesture_mouse as gm

cap = gm._open_camera("auto")
assert cap is not None, "摄像头不可用"
face = gm.HeadPoseEngine()
hands, _ = gm.make_engine("mediapipe")

def bench(name, run_hands_every):
    n = 0
    t0 = time.perf_counter()
    t_face = 0.0
    t_hand = 0.0
    while n < 90:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        a = time.perf_counter()
        face.detect(rgb)
        t_face += time.perf_counter() - a
        if n % run_hands_every == 0:
            a = time.perf_counter()
            hands.detect(rgb, touch_threshold=0.55)
            t_hand += time.perf_counter() - a
        n += 1
    dt = time.perf_counter() - t0
    print("%s: %.1f fps | 人脸 %.1f ms/帧 | 手部 %.1f ms/次(每%d帧)"
          % (name, n / dt, t_face / n * 1000, t_hand / (n / run_hands_every) * 1000,
             run_hands_every))

bench("旧方案(人脸+手部每帧串行)", 1)
bench("新方案(手部降频 1/3)", 3)
face.close()
hands.close()
cap.release()
print("BENCH DONE")
