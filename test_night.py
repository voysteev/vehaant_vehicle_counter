from night_tail_light_tracking import VisualizedSolution
import os

# --- CONFIGURATION ---
video_path = "videos/vehant_hackathon_video_1.avi"  
output_filename = "output_result.mp4"

def run():
    print(f"Loading video from: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"ERROR: File not found at {video_path}")
        return

    solution = VisualizedSolution()
    
    # Enable saving by providing output_path
    solution.forward(
        video_path, 
        output_path=output_filename, 
        display_live=True
    )
    
    print("Done! You can now upload 'output_result.mp4'.")

if __name__ == "__main__":
    run()