# VisionCheckout-RPC 

An automated product recognition system designed to simulate the retail checkout process. This project utilizes the **Retail Product Checkout (RPC)** dataset to identify multiple items in a single frame, providing a foundation for "just-walk-out" shopping technology.

## Overview
The goal of this project is to build a robust computer vision pipeline that can:
1. Detect multiple products in a retail environment.
2. Classify items based on the RPC dataset categories.
3. Simulate a checkout receipt based on recognized items.

## Dataset
We use the **[RPC (Retail Product Checkout) Dataset](https://rpc-dataset.github.io/)**, which contains:
* **200** product categories.
* Thousands of high-resolution images across different lighting and clutter levels.
* Single-item (exemplar) and multi-item (checkout) images.

## Tech Stack
* **Language:** Python 3.11.9 (this Version Specifically to avoid compatibality issues)
* **Frameworks:** PyTorch 
* **Libraries:** OpenCV, NumPy, Matplotlib
* **Version Control:** Git & GitHub

## ⚙️ Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abdullah-Ghulam/Vision-Checkout
    cd VisionCheckout
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

4.  **Verify Installation**  
After installing the requirements,
run the following script to ensure your environment (Python, PyTorch, and GPU) is configured correctly:

```bash
python check_setup.py
```

## Project Structure
```text
Vision_Checkout
├── data/               # Local dataset (Ignored by Git)
|   ├──── train2019
|   ├──── val2019
|   ├──── test2019
|   ├──── instances_train2019.json
|   ├──── instances_val2019.json
|   ├──── instances_test2019.json
|
├── venv/ or .env       # Python Environemnt (Ignore by Git)
├── models/             # Saved model weights (.pth or .h5)
├── notebooks/          # EDA and prototyping
├── src/                # Source code for training and inference
├── .gitignore          # Files to exclude from Git
└── README.md
