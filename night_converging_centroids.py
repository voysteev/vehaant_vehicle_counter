import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class VehicleCounterVisualized:
    """
    FINAL VERSION: MAX SENSITIVITY + CENTROID MERGING
    """

    def __init__(self):
        self.tracks = {}
        self.track_seq = 0
        self.counted = set()
        
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

        # Kernel to make sure individual lights are solid dots
        self.k_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    def process(self, frame):
        if frame.shape[1] != 1280:
            aspect = frame.shape[0] / frame.shape[1]
            frame = cv2.resize(frame, (1280, int(1280 * aspect)))
        return frame

    def get_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # FIX 1: HIGH SENSITIVITY
        # Lowered to 50/50.Catches very dim lights.
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        
        lower_red2 = np.array([165, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        motion_mask = self.fgbg.apply(frame)
        _, motion_mask = cv2.threshold(motion_mask, 200, 255, cv2.THRESH_BINARY)
        
        mask = cv2.bitwise_and(red_mask, motion_mask)
        
        # Fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_morph, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k_morph, iterations=1)
        
        return mask

    def get_dets(self, mask, h, w):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_candidates = []
        min_a = 30   
        max_a = (w * h * 0.6)

        # 1. Gather all raw contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_a or area > max_a:
                continue

            x, y, wb, hb = cv2.boundingRect(cnt)
            
            if y < h * 0.05 or y > h * 0.95:
                continue
            
            ratio = wb / float(hb)
            if ratio < 0.3: 
                continue
            
            cx = x + wb // 2
            cy = y + hb // 2
            
            raw_candidates.append({
                "c": (cx, cy),
                "bbox": (x, y, wb, hb),
                "a": area,
                "merged": False
            })

        # FIX 2: MERGE NEARBY CENTROIDS
        # If two lights are close, they become ONE detection.
        final_dets = []
        
        # Sort by X to make finding neighbors easier
        raw_candidates.sort(key=lambda c: c["c"][0])
        
        for i in range(len(raw_candidates)):
            if raw_candidates[i]["merged"]:
                continue
            
            c1 = raw_candidates[i]
            
            # Start with this centroid
            final_cx = c1["c"][0]
            final_cy = c1["c"][1]
            final_area = c1["a"]
            count_merged = 1
            
            # Look ahead for neighbors
            for j in range(i + 1, len(raw_candidates)):
                if raw_candidates[j]["merged"]:
                    continue
                
                c2 = raw_candidates[j]
                
                dx = abs(c1["c"][0] - c2["c"][0])
                dy = abs(c1["c"][1] - c2["c"][1])
                
                # MERGE CRITERIA:
                # Close Horizontal (<120px) AND Same Height (<40px)
                if dx < 120 and dy < 120:
                    # It's a match! Merge it.
                    final_cx += c2["c"][0]
                    final_cy += c2["c"][1]
                    final_area += c2["a"]
                    count_merged += 1
                    raw_candidates[j]["merged"] = True # Mark used
            
            # Average the position
            avg_cx = int(final_cx / count_merged)
            avg_cy = int(final_cy / count_merged)
            
            final_dets.append({
                "c": (avg_cx, avg_cy),
                "bbox": c1["bbox"], # Just keep original bbox for ref
                "a": final_area
            })

        return final_dets

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
                self.tracks[tid]["lost"] = 0
                self.tracks[tid]["is_dup"] = False 
                matched_t.add(tid)
                matched_d.add(c)

        for tid in t_ids:
            if tid not in matched_t:
                self.kill_track(tid)

        for j, d in enumerate(dets):
            if j not in matched_d:
                self.add_track(d)
                
        # Still keep vertical pruning for shadows
        self.prune_tracks()
        
        for tid in list(self.tracks.keys()):
            self.check_count(tid, h)

    def prune_tracks(self):
        ids = list(self.tracks.keys())
        for i in range(len(ids)):
            id_a = ids[i]
            if self.tracks[id_a]["lost"] > 0: continue
            pos_a = self.tracks[id_a]["pts"][-1]
            
            for j in range(i + 1, len(ids)):
                id_b = ids[j]
                if self.tracks[id_b]["lost"] > 0: continue
                pos_b = self.tracks[id_b]["pts"][-1]
                
                dx = abs(pos_a[0] - pos_b[0])
                dy = abs(pos_a[1] - pos_b[1])
                
                # Rule: Remove LOWER shadow (Vertical Pair)
                if dx < 30 and dy < 80:
                    if pos_a[1] > pos_b[1]:
                        self.tracks[id_a]["is_dup"] = True
                    else:
                        self.tracks[id_b]["is_dup"] = True

    def add_track(self, d):
        self.tracks[self.track_seq] = {
            "pts": deque([d["c"]], maxlen=50),
            "lost": 0,
            "done": False,
            "is_dup": False
        }
        self.track_seq += 1

    def kill_track(self, tid):
        self.tracks[tid]["lost"] += 1
        if self.tracks[tid]["lost"] > 10:
            del self.tracks[tid]

    def check_count(self, tid, h):
        t = self.tracks[tid]
        if t["is_dup"] or t["done"]: return
        if len(t["pts"]) < 4: return

        y0 = t["pts"][0][1]
        y1 = t["pts"][-1][1]
        
        if y1 > y0: return

        line = h * 0.40
        if y1 < line and y0 > line:
            self.counted.add(tid)
            self.tracks[tid]["done"] = True
            print(f" ✓ Vehicle counted: Track {tid}")

    def result(self):
        return len(self.counted)

    def visualize(self, original_frame, processed_frame, mask, dets, h):
        target_h, target_w = processed_frame.shape[:2]

        panel1 = cv2.resize(original_frame, (target_w, target_h))
        line_y = int(h * 0.40)
        cv2.line(panel1, (0, line_y), (panel1.shape[1], line_y), (0, 255, 0), 2)
        cv2.putText(panel1, "1. INPUT", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        panel2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "2. MASK (SENSITIVE)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        panel3 = cv2.resize(original_frame.copy(), (target_w, target_h))
        cv2.line(panel3, (0, line_y), (panel3.shape[1], line_y), (0, 255, 0), 1)
        
        # Draw detections as BLUE dots
        for d in dets:
            cv2.circle(panel3, d["c"], 4, (255, 0, 0), -1)

        for tid, track in self.tracks.items():
            if track["lost"] > 0: continue
            
            pts = list(track["pts"])
            if len(pts) > 0:
                cx, cy = pts[-1]
                
                if track["is_dup"]:
                    color = (0, 0, 255)
                    status = "X"
                else:
                    color = (0, 255, 0)
                    status = str(tid)
                    
                if tid in self.counted:
                    color = (255, 255, 0)
                
                cv2.circle(panel3, (cx, cy), 6, color, -1)
                cv2.putText(panel3, status, (cx-5, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(panel3, "3. MERGED TRACKS", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        panel4 = np.zeros_like(panel1)
        count_text = f"COUNT: {len(self.counted)}"
        cv2.putText(panel4, count_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)

        top = np.hstack([panel1, panel2])
        btm = np.hstack([panel3, panel4])
        combined = np.vstack([top, btm])
        
        return cv2.resize(combined, (combined.shape[1]//2, combined.shape[0]//2))


# ==========================================
# 2. THE RUNNER CLASS
# ==========================================
class VisualizedSolution:
    def forward(self, video_path: str, output_path: str = None, display_live: bool = True):
        vc = VehicleCounterVisualized()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video at {video_path}")
            return 0

        print(f"Processing {video_path}...")
        
        writer = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            p = vc.process(frame)
            h, w = p.shape[:2]
            mask = vc.get_mask(p)
            dets = vc.get_dets(mask, h, w)
            vc.update(dets, h)

            vis = vc.visualize(frame, p, mask, dets, h)

            if output_path:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(output_path, fourcc, 30, (vis.shape[1], vis.shape[0]))
                
                writer.write(vis)

            if display_live:
                cv2.imshow('Taillight Counter', vis)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        cap.release()
        if writer:
            writer.release()
            print(f"Video saved to: {output_path}")
            
        cv2.destroyAllWindows()
        return vc.result()