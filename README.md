# Deep Learning

This repository documents my Deep Learning learning journey through hands-on implementations of fundamental concepts and advanced computer vision techniques using TensorFlow/Keras and PyTorch.

The repository follows a structured roadmap starting from the basic Perceptron model and gradually progressing towards Artificial Neural Networks (ANNs), Convolutional Neural Networks (CNNs), advanced CNN architectures, Object Detection, Image Segmentation, Siamese Networks, and Generative Adversarial Networks (GANs).

---

## Repository Highlights

- 35+ Jupyter notebooks covering core Deep Learning concepts
- Implementations using both TensorFlow/Keras and PyTorch
- Neural Networks built from basic concepts to advanced architectures
- CNN architectures including LeNet, AlexNet, VGG, Inception, ResNet, and Xception
- Transfer Learning using pre-trained models
- Object Detection using R-CNN and Faster R-CNN (Detectron2)
- Image Segmentation using Mask R-CNN
- Face Recognition using Siamese Networks
- Generative modeling using GANs and DCGANs

---

## Learning Roadmap

| Module | Topics Covered |
|--------|----------------|
| 01. Perceptron | Single Perceptron, XOR Problem |
| 02. Activation Functions | Sigmoid, Tanh, ReLU and Non-linear Transformations |
| 03. Deep Learning Frameworks | TensorFlow and PyTorch Fundamentals, MNIST and Fashion-MNIST |
| 04. Artificial Neural Networks | ANN implementation using TensorFlow and PyTorch |
| 05. CNN Basics | Image Processing Filters, CNN on MNIST and CIFAR-10 |
| 06. Classic CNN Architectures | LeNet, AlexNet, VGG, Inception, ResNet, Xception, YOLO-based Classification |
| 07. Object Detection | Fire & Smoke Detection, R-CNN, Faster R-CNN using Detectron2 |
| 08. Image Segmentation | Mask R-CNN implementation and custom dataset training |
| 09. Siamese Networks | Face Recognition using Siamese Architecture |
| 10. Generative Models | GAN from Scratch and DCGAN implementation |

---

## Repository Structure

```text
Deep_Learning/
│
├── README.md
├── requirements.txt
│
├── 01_Perceptron/
├── 02_Activation_Functions/
├── 03_Deep_Learning_Frameworks/
│   ├── TensorFlow/
│   └── PyTorch/
│
├── 04_Artificial_Neural_Networks/
├── 05_CNN_Basics/
├── 06_Classic_CNN_Architectures/
├── 07_Object_Detection/
├── 08_Image_Segmentation/
├── 09_Siamese_Networks/
└── 10_Generative_Models/
```

---

## Technologies & Libraries Used

- Python
- TensorFlow / Keras
- PyTorch & Torchvision
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

---

## Project Purpose

The purpose of this repository is to build a strong practical understanding of Deep Learning by implementing models from scratch, experimenting with different architectures, and learning how modern computer vision systems are developed.

Each notebook contains experiments, implementations, and observations that helped me understand the underlying concepts of Deep Learning.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-github-repository-link>
cd Deep_Learning
```

### 2. Create a Virtual Environment

**Windows (PowerShell)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter Notebook

```bash
jupyter lab
```

Open any notebook from the corresponding module folder and explore the implementations.

---

## Future Scope

This repository will continue to evolve with additional Deep Learning concepts and more advanced computer vision projects, including:

- U-Net based image segmentation
- Advanced GAN architectures
- Diffusion models
- More real-world computer vision applications

---

## License

This repository is created for learning and educational purposes. Please refer to individual datasets and pre-trained models for their respective licenses.