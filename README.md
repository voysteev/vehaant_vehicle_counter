# Vehicle Counting Solution - Vehant Hackathon
To run:
    python3 test_solution.py ./pathto/file_name

## 1. Methodology & Approach
This solution implements a robust, classical computer vision pipeline designed to count vehicles from a static camera without using Deep Learning. The core philosophy is **"Adaptive filtering over static detection."**

Instead of treating every moving pixel as a vehicle, the system uses a multi-stage filter to distinguish between **noise** (fog, rain, shadows) and **valid traffic** (cars, autos, motorbikes).

### Processing Pipeline
1.  **Background Modeling (MOG2):**
    * We use a Gaussian Mixture-based Background/Foreground Segmentation Algorithm (MOG2).
    * **Configuration:** `history=500`, `varThreshold=30`. This specific tuning is sensitive enough to detect dark vehicles in low contrast but robust enough to ignore gradual lighting changes.

2.  **Morphological Processing ("The Vertical Cut"):**
    * **Problem:** On highways, tailgating vehicles often merge into a single "blob," causing undercounting.
    * **Solution:** We apply a custom **(1, 5) vertical erosion kernel**. This physically slices the thin connections between vehicles moving in a line, forcing the system to recognize them as distinct objects.

3.  **Adaptive Stability Filter (Dual-Gate Logic):**
    * We observed that "noise" (fog ghosts) and "signal" (vehicles) behave differently in terms of size and persistence. We devised a dual-threshold logic:
        * **Large Objects (Area > 3800):** Likely Cars/Autos. These are counted quickly (after **5 frames**) to capture fast-moving traffic.
        * **Small Objects (Area < 3800):** Likely Motorbikes or Fog Noise. We require these to persist for **24 frames**. This duration effectively filters out weather noise (which typically flickers for <18 frames) while preserving valid motorbikes.

4.  **Counting Mechanism:**
    * A virtual line is established at **60% of the screen height**.
    * Vehicles are tracked using a centroid-based tracker with Hungarian Algorithm assignment.
    * A count is valid only if the object crosses the line moving upwards (away from the camera).

---

## 2. Key Design Choices

| Challenge | Design Choice | Rationale |
| :--- | :--- | :--- |
| **Fog & Weather Noise** | **High Stability Threshold (24 frames)** | Fog "ghosts" are transient. By forcing small objects to prove their stability over ~1 second, we eliminate false positives without reducing sensitivity for real cars. |
| **Tailgating Traffic** | **Vertical Kernel Erosion** | Standard erosion shrinks objects uniformly. Vertical erosion specifically targets the "neck" between two cars, separating them while preserving their width. |
| **Mixed Traffic (Autos)** | **Area Threshold (3800)** | Auto-rickshaws are smaller than cars but larger than bikes. Lowering the "Large Object" threshold to 3800 ensures they are not mistaken for noise. |
| **Hardware Constraints** | **CPU-Only OpenCV** | The solution uses lightweight matrix operations (NumPy/OpenCV), ensuring it runs efficiently on standard hardware without GPU acceleration. |

---

## 3. Assumptions
1.  **Static Camera:** The background subtraction relies on the camera position remaining fixed.
2.  **Directionality:** The logic assumes vehicles are primarily moving away from the camera (bottom-to-top flow).
3.  **Frame Rate:** The stability metrics (e.g., 24 frames) assume a standard video frame rate (approx. 25-30 fps).
4.  **Lighting:** While MOG2 handles illumination changes, the system assumes there are no extreme, instantaneous lighting strobes (e.g., lightning) that would flood the entire sensor.

---

## 4. Setup & Execution

### Prerequisites
* Python 3.x
* Standard libraries: `opencv-python`, `numpy`, `scipy`

### Installation
```bash
pip install -r requirements.txt