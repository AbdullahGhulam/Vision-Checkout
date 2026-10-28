# Vision Checkout

An automated self-checkout system that detects and classifies retail products from a single image and generates a receipt automatically. Built on the **Retail Product Checkout (RPC)** dataset and fine-tuned on a **custom Saudi retail dataset** to perform well on local products.

🔗 **Live Demo:** [Hugging Face Space](https://huggingface.co/spaces/Almo7tarif/vision-checkout)

---

## Overview
Standard detectors struggle with visually similar products, occlusion, and varying lighting at checkout counters, and global datasets don't represent Saudi products. This project builds a complete pipeline that:
1. Detects multiple products in a single image.
2. Classifies them into 17 retail categories adapted to the Saudi market.
3. Generates a digital receipt with quantities, prices, and VAT.

## Datasets
| Dataset | Description |
|---|---|
| [RPC Dataset](https://rpc-dataset.github.io/) | 80,000+ images, 200 product classes. Used for initial training. |
| Saudi Local Dataset | 857 images of local Saudi brands, **manually annotated** using Roboflow. Used for fine-tuning. |

## Methodology
**1. Data Engineering**
- Converted annotations from COCO to YOLO format.
- Applied augmentations (Gaussian Blur, Brightness/Contrast, Mosaic, CoarseDropout, Mixup, HSV) to simulate real checkout conditions: blur, harsh lighting, and occlusion.

**2. Two-Stage Training with YOLO26m**
- **Stage 1:** Trained on the RPC dataset to build a strong feature-extraction backbone.
- **Stage 2:** Fine-tuned on a hybrid dataset (RPC subset + Saudi data), remapping 200 classes into 17, freezing the backbone (`freeze=8`), and using oversampling to handle class imbalance.

**3. Training Environment**
- Google Colab Enterprise with NVIDIA RTX PRO 6000 (172GB RAM).
- AdamW optimizer, batch size 128, image size 640, early stopping (patience=20).

## Results
| Model | mAP50 | Precision | Recall | F1 |
|---|---|---|---|---|
| YOLO11 (baseline) | 37.0% | 39.0% | 43.1% | 40.9% |
| **YOLO26m (fine-tuned)** | **46.4%** | **61.8%** | 41.8% | **49.9%** |

### Bounding Box Detection
![Bounding Boxes](assests/bounding_boxes.png)

### Data Augmentation Examples
![Data Augmentation](assests/Augmentation.jpeg)

### Labeled Products
![Labels](assests/lables.jpg)

## Deployment
Deployed on **Hugging Face Spaces** using **Gradio** (Streamlit was tested first but degraded inference quality). Features:
- Image upload or random sample selection
- Adjustable confidence threshold
- Real-time detection with bounding boxes
- Automatic receipt generation with price mapping and VAT

## Limitations
- Accuracy drops when products are heavily stacked or overlapping.
- 17 broad categories cannot yet distinguish between brands of the same product type.

## Tech Stack
* **Language:** Python 3.11.9
* **Deep Learning:** PyTorch, Ultralytics YOLO26
* **Data:** Albumentations, OpenCV, Pandas, NumPy, Matplotlib, Roboflow
* **Deployment:** Gradio, Hugging Face Spaces
* **Tools:** Git, GitHub, Google Colab Enterprise

## Setup & Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/Abdullah-Ghulam/Vision-Checkout
    cd Vision-Checkout
    ```
2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Verify installation (Python, PyTorch, GPU):**
    ```bash
    python check_setup.py
    ```

## Project Structure
```text
Vision_Checkout
├── data/               # Local dataset and annotations (Ignored by Git)
│   ├── instances_test2019.json
│   ├── instances_train2019.json
│   ├── instances_val2019.json
│   ├── test2019/       
│   ├── train2019/      
│   └── val2019/        
├── venv/               # Isolated Python environment (Ignored by Git)
├── models/             # Saved model weights (.pt, .onnx) and checkpoints
├── notebooks/          # Jupyter notebooks for EDA and model prototyping
│  └── 01_EDA.ipynb
├── src/                # Core source code 
├── requirements.txt    # List of project dependencies and libraries
├── check_setup.py      # Script to verify CUDA and environment configuration
└── README.md           # Project documentation and setup instructions
```

