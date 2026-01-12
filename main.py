import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class VehicleCounter:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=30, detectShadows=True)
        self.tracks = {}
        self.track_seq = 0
        self.counted = set()
        
        self.k_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        self.k_restore = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame):
        if frame.shape[1] != 1280:
            aspect = frame.shape[0] / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(1280 * aspect)))
        return cv2.GaussianBlur(frame, (7, 7), 0)
        
    def get_mask(self, frame):
        mask = self.fgbg.apply(frame, learningRate=0.001)
        _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_noise, iterations=1)
        mask = cv2.erode(mask, self.k_vert, iterations=1)
        mask = cv2.dilate(mask, self.k_restore, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_restore, iterations=1)
        return mask
        
    def get_dets(self, mask, h, w):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        
        min_a = 450 
        max_a = (w * h) * 0.60
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue
                
            x, y, wb, hb = cv2.boundingRect(cnt)
            ratio = wb / float(hb)
            
            if ratio < 0.15 or ratio > 5.0:
                continue
            
            if y > h * 0.98 or y < h * 0.05:
                continue

            cx = x + wb // 2
            cy = y + hb // 2
            
            dets.append({
                "c": (cx, cy),
                "bbox": (x, y, wb, hb),
                "a": area
            })
            
        return dets
        
    def update(self, dets, h):
        t_ids = list(self.tracks.keys())
        
        if len(self.tracks) == 0:
            for d in dets:
                self.add_track(d)
            return

        if len(dets) == 0:
            for tid in t_ids:
                self.kill_track(tid)
            return

        costs = np.zeros((len(t_ids), len(dets)))
        for i, tid in enumerate(t_ids):
            last = self.tracks[tid]["pts"][-1]
            for j, d in enumerate(dets):
                dist = np.sqrt((last[0] - d["c"][0])**2 + (last[1] - d["c"][1])**2)
                costs[i, j] = dist
        
        rows, cols = linear_sum_assignment(costs)
        
        matched_t = set()
        matched_d = set()
        
        thresh = h * 0.20
        
        for r, c in zip(rows, cols):
            if costs[r, c] < thresh:
                tid = t_ids[r]
                self.tracks[tid]["pts"].append(dets[c]["c"])
                self.tracks[tid]["max_a"] = max(self.tracks[tid]["max_a"], dets[c]["a"])
                self.tracks[tid]["lost"] = 0
                
                self.check_count(tid, h)
                
                matched_t.add(tid)
                matched_d.add(c)
        
        for tid in t_ids:
            if tid not in matched_t:
                self.kill_track(tid)
        
        for j, d in enumerate(dets):
            if j not in matched_d:
                self.add_track(d)

    def add_track(self, d):
        self.tracks[self.track_seq] = {
            "pts": deque([d["c"]], maxlen=100),
            "max_a": d["a"], 
            "lost": 0,
            "done": False
        }
        self.track_seq += 1

    def kill_track(self, tid):
        self.tracks[tid]["lost"] += 1
        if self.tracks[tid]["lost"] > 20: 
            del self.tracks[tid]

    def check_count(self, tid, h):
        if self.tracks[tid]["done"]:
            return

        t = self.tracks[tid]
        
        req = 24
        if t["max_a"] > 3800:
            req = 5
            
        if len(t["pts"]) < req:
            return
            
        y0 = t["pts"][0][1]
        y1 = t["pts"][-1][1]
        
        if y1 >= y0:
            return

        line = h * 0.60
        
        if y1 < line and y0 > line:
            dist = 50
            if t["max_a"] > 3800:
                dist = 30 
                
            if (y0 - y1) > dist:
                self.counted.add(tid)
                self.tracks[tid]["done"] = True
                print(f"  ✓ Vehicle counted: Track {tid} (Moved {y0 - y1:.0f}px UP, Frames: {len(t['pts'])}, MaxArea: {t['max_a']:.0f})")

    def result(self):
        return len(self.counted)

class Solution:
    def forward(self, video_path: str) -> int:
        vc = VehicleCounter()
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error: Cannot open video")
            return 0
        
        for _ in range(10):
            ret, frame = cap.read()
            if not ret: break
            p = vc.process(frame)
            vc.fgbg.apply(p, learningRate=0.5)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {video_path}")
        print(f"Total frames: {tot}")
        print("-" * 50)
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            idx += 1
            p = vc.process(frame)
            h, w = p.shape[:2]
            
            mask = vc.get_mask(p)
            dets = vc.get_dets(mask, h, w)
            vc.update(dets, h)
            
            if idx % 200 == 0:
                 print(f"  Frame {idx}/{tot} | Active Tracks: {len(vc.tracks)} | Counted: {len(vc.counted)}")
            
        cap.release()
        
        final = vc.result()
        print("-" * 50)
        print(f"Processing complete!")
        print(f"Total tracks created: {vc.track_seq}")
        print(f"FINAL COUNT: {final} vehicles")
        print("=" * 50)
        
        return final