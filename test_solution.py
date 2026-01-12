import sys
from main import Solution

def test_video(video_path):
    print(f"Testing video: {video_path}")
    print("-" * 50)
    
    solution = Solution()
    count = solution.forward(video_path)
    
    print("-" * 50)
    print(f"Final Result: {count} vehicles")
    return count

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_solution.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_video(video_path)

