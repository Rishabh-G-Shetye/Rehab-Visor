# Rehab-Visor 🏥🤖
> **AI-Powered Virtual Physical Therapy Coach** > *1st Place Winner - Lenovo LEAP AI Hackathon 2026* 🏆

Rehab-Visor is a computer vision-based application that acts as a "Smart Mirror" for physical therapy. It uses Edge AI to track body movements via a standard webcam, providing real-time corrective feedback and gamified rep counting to ensure patients perform exercises correctly at home.

---

## 🚀 Features
* **Real-Time Skeleton Tracking:** Uses MediaPipe Pose to detect 33 body landmarks instantly.
* **AI Form Correction:** Calculates joint angles (e.g., elbow flexion, knee depth) to judge exercise quality.
    * *Green Feedback:* Perfect form (Rep counted).
    * *Red Feedback:* Incorrect form (Audio/Visual alert).
* **Gamified "Ghost" Mode:** Overlay a perfect reference skeleton for users to mimic.
* **Privacy-First:** Runs 100% locally on-device (Edge AI). No video data is sent to the cloud.
* **Session Analytics:** Exports performance data to CSV for tracking progress.

## 🛠️ Tech Stack
* **Language:** Python 3.11
* **Core AI:** Google MediaPipe (Pose Estimation)
* **Visualization:** OpenCV & NumPy (Trigonometry for angle calculation)
* **Frontend/UI:** Streamlit
* **Hardware:** Optimized for Lenovo Laptops & Tablets (Runs on CPU)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/Rehab-Visor.git](https://github.com/yourusername/Rehab-Visor.git)
    cd Rehab-Visor
    ```

2.  **Create a Virtual Environment (Recommended: Python 3.11)**
    ```bash
    py -3.11 -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

Run the application using Streamlit:
```bash
streamlit run rehab_ghost.py

