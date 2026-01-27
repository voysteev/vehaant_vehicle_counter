import cv2
import os
import sys
import numpy as np

# --- IMPORT YOUR MODULES ---
try:
    # Day Mode Logic
    from main import Solution as DaySolution
    
    # Night Mode Logic
    from night_converging_centroids import VisualizedSolution as NightSolution
    
except ImportError as e:
    print("CRITICAL ERROR: Could not import required files.")
    print(f"Details: {e}")
    print("Ensure 'main_visualized.py' and 'main3.py' are in this folder.")
    sys.exit(1)

class UnifiedDriver:
    def is_night_video(self, video_path):
        """
        Analyzes the first 30 frames to determine Day vs Night.
        Uses TWO metrics for robustness against bright streetlights.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot read video for analysis: {video_path}")
            return False
        
        brightness_samples = []
        dark_ratio_samples = []
        
        for _ in range(30):
            ret, frame = cap.read()
            if not ret: break
            
            # Convert to HSV to check 'Value' (Brightness) channel
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            v_channel = hsv[..., 2]
            
            # Metric 1: Average Brightness (0-255)
            avg_b = v_channel.mean()
            
            # Metric 2: Dark Pixel Ratio (0.0 - 1.0)
            # Count how many pixels are darker than 60 (Shadows/Night Sky)
            dark_pixels = np.count_nonzero(v_channel < 60)
            total_pixels = v_channel.size
            dark_ratio = dark_pixels / total_pixels
            
            brightness_samples.append(avg_b)
            dark_ratio_samples.append(dark_ratio)
            
        cap.release()
        
        if not brightness_samples:
            return False
            
        final_avg_b = sum(brightness_samples) / len(brightness_samples)
        final_dark_r = sum(dark_ratio_samples) / len(dark_ratio_samples)
        
        print(f"[*] ANALYSIS STATS:")
        print(f"    Avg Brightness: {final_avg_b:.2f} (Day > 110)")
        print(f"    Dark Pixel %:   {final_dark_r*100:.1f}% (Night > 40%)")
        
        # --- THE DECISION LOGIC ---
        # It is NIGHT if:
        # 1. The image is genuinely dark (Avg < 100)
        # OR
        # 2. The image is "High Contrast" (Avg is high due to lights, but >40% is black)
        
        is_night = (final_avg_b < 110) or (final_dark_r > 0.40)
        
        return is_night

    def run(self, video_path, output_filename="output.mp4"):
        if not os.path.exists(video_path):
            print(f"Error: File not found -> {video_path}")
            return

        # 1. DECIDE MODE
        print("-" * 40)
        print(f"Analyzing Video: {video_path}")
        is_night = self.is_night_video(video_path)
        
        # 2. RUN APPROPRIATE FILE
        if is_night:
            print(">>> RESULT: Nighttime Detected.")
            print(">>> LOADING: main3.py (Red Sensitivity Logic)")
            print("-" * 40)
            solver = NightSolution()
        else:
            print(">>> RESULT: Daytime Detected.")
            print(">>> LOADING: main_visualized.py (Shadow Removal Logic)")
            print("-" * 40)
            solver = DaySolution()

        # 3. EXECUTE
        solver.forward(
            video_path, 
            output_path=output_filename, 
            display_live=True
        )

if __name__ == "__main__":
    # CONFIGURATION
    VIDEO_FILE = "videos/vehant_hackathon_video_8.avi" 
    OUTPUT_FILE = "final_output.mp4"
    
    driver = UnifiedDriver()
    driver.run(VIDEO_FILE, OUTPUT_FILE)