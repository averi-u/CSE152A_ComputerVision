# CSE152A: Computer Vision (UCSD)

This repository contains projects from **CSE152A - Computer Vision** uploaded after graduation as public. Single Author: Zhizhen Averi Yu. 
Each assignment explores fundamental topics in computer vision, from image formation to 3D reconstruction, implemented in **Python** and **Jupyter Notebooks**.
![HW4 CNN Results](HW4/output_digit_classification.png)

---

## 📁 Repository Structure

│
├── HW0/ # Intro to image manipulation and pixel operations  
├── HW1/ # Edge detection and filtering  
├── HW2/ # Feature detection and matching  
├── HW3/ # Camera calibration and 3D reconstruction  
├── HW4/ # Deep learning-based image classification  



---

## 🧠 Homework Summaries

### 🧩 **HW0 — Image Manipulation Basics**
**Topics:**  
- Image I/O, color spaces (RGB, grayscale)  
- Pixel operations and transformations  
- Histogram equalization and normalization  

**Highlights:**  
- Learned to manipulate and visualize images using NumPy and OpenCV  
- Implemented basic geometric transformations (flip, crop, rotate)

---

### ⚙️ **HW1 — Edge Detection & Filtering**
**Topics:**  
- Image filtering (Gaussian, Sobel, Laplacian)  
- Gradient magnitude and orientation  
- Canny edge detector  

**Highlights:**  
- Implemented edge detection from scratch  
- Compared results with built-in OpenCV edge detection functions  

---

### 🔍 **HW2 — Feature Detection & Matching**
**Topics:**  
- Harris corner detection  
- SIFT/ORB feature extraction  
- Feature matching using SSD/NCC  

**Highlights:**  
- Built a simple image stitching pipeline  
- Visualized keypoints and matches between overlapping images  

---

### 📷 **HW3 — Camera Calibration & 3D Reconstruction**
**Topics:**  
- Image classfication 
- Feature Extraction
- Pinhole camera model and projection matrices  
- Epipolar geometry, essential & fundamental matrices  
- Triangulation for 3D point reconstruction  

**Highlights:**  
- Computed camera intrinsics and extrinsics  
- Reconstructed 3D scenes from stereo image pairs  

---

### 🤖 **HW4 — Deep Learning for Vision Tasks**
**Topics:**  
- CNN fundamentals (convolution, pooling, activation)  
- Training and testing a simple image classifier for digit classification
- Transfer learning with pre-trained models  

**Highlights:**  
- Built and trained CNNs using PyTorch  
- Achieved >90% accuracy on a small image dataset  
- Visualized feature maps and learned filters  

---

## 🧰 Tech Stack

| Category | Tools / Libraries | Purpose |
|-----------|-------------------|----------|
| **Language** | Python 3.10 | Core programming language |
| **Numerical Computing** | NumPy, SciPy | Matrix operations, linear algebra |
| **Image Processing** | OpenCV, PIL | I/O, color conversion, filtering |
| **Visualization** | Matplotlib, Seaborn | Image plots, histograms, metrics |
| **Feature Detection** | OpenCV (SIFT, ORB, Harris) | Keypoint extraction and matching |
| **3D Geometry** | OpenCV Calibration, NumPy | Camera models, triangulation |
| **Deep Learning** | PyTorch, TorchVision | CNNs and transfer learning |
| **Model Architectures Used** | Custom CNN, ResNet18, VGG16 | Image classification |
| **Environment** | Jupyter Notebook | Interactive experimentation |
| **Version Control** | Git, GitHub | Code management and collaboration |

---

## 🧮 Mathematical Foundations
- Convolution & Correlation  
- Gradient-based edge detection  
- Homography estimation using RANSAC  
- Camera projection equations  
- Cross-entropy loss and backpropagation  

---

## 📸 Example Outputs


![HW1 Edge Detection](HW1/Problem1_example.png)



| Visualization | Output |
|----------------|---------|
| **Corner Detection** | ![HW2 Corner Detection](HW2/output_images/output_corner_detection.png) |
| **Edge Detection** | ![HW2 Edge Detection](HW2/output_images/output_edgeDetection.png) |
| **Epipolar Lines** | ![HW2 Epipolar Lines](HW2/output_images/output_epipolarLines.png) |
| **Naive Feature Matching** | ![HW2 Naive Matching](HW2/output_images/outout_naive_matching.png) |


![HW3 3D Reconstruction](HW3/output_image_classfication.png)

