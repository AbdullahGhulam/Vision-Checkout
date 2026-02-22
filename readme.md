# VisionCheckout-RPC 🛒

An automated product recognition system designed to simulate the retail checkout process. This project utilizes the **Retail Product Checkout (RPC)** dataset to identify multiple items in a single frame, providing a foundation for "just-walk-out" shopping technology.

## Overview
The goal of this project is to build a robust computer vision pipeline that can:
1. Detect multiple products in a retail environment.
2. Classify items based on the RPC dataset categories.
3. Simulate a checkout receipt based on recognized items.

## Dataset
We use the **RPC (Retail Product Checkout) Dataset**, which contains:
* **200** product categories.
* Thousands of high-resolution images across different lighting and clutter levels.
* Single-item (exemplar) and multi-item (checkout) images.

## Tech Stack
* **Language:** Python 3.10+
* **Frameworks:** PyTorch / TensorFlow (Select one)
* **Libraries:** OpenCV, NumPy, Matplotlib
* **Version Control:** Git & GitHub

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abdullah-Ghulam/Vision-Checkout
    cd VisionCheckout-RPC
    ```

2.  **Create a Virtual Environment:**
    ```bash
    #this creates a new environment (run it only ONCE)
    python -m venv venv

    #after that activate the environement using:
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 Project Structure
```text
├── data/               # Local dataset (Ignored by Git)
├── models/             # Saved model weights (.pth or .h5)
├── notebooks/          # EDA and prototyping
├── src/                # Source code for training and inference
├── .gitignore          # Files to exclude from Git
└── README.md