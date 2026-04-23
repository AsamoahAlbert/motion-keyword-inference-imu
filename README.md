# Motion Sensor-Based Keyword Inference (IMU)

> CNN-based system that infers spoken keywords from IMU vibration data and reveals IoT side-channel privacy risks.

---

## Overview
This project investigates whether vibration signals captured by motion sensors can be used to infer spoken keywords without directly accessing audio.

An ADXL345 accelerometer is used to capture vibration data generated during audio playback. A 1D Convolutional Neural Network (CNN) is trained to classify these vibration signals into predefined keyword categories.

## Key Takeaway
This work demonstrates that motion sensors can act as **unintended side channels**, potentially leaking speech-related information.  
While the approach is feasible under controlled conditions, it highlights **significant generalization challenges** and **privacy risks in IoT systems**.

---

## Objectives
- Evaluate feasibility of vibration-based keyword inference  
- Compare performance in controlled vs real-world environments  
- Analyze privacy risks from sensor-based side-channel leakage  

---

## System Pipeline
1. Keyword audio playback via speaker  
2. Vibration capture using ADXL345 sensor  
3. Time-series preprocessing (normalization, segmentation)  
4. CNN-based classification  
5. Evaluation on dataset and real-world recordings  

---

## Dataset
- ~13,000 labeled IMU samples (structured dataset)  
- 100 real-world samples collected using Raspberry Pi + ADXL345  
- Sampling rate: 200 Hz  
- Input format: 3-axis accelerometer data (x, y, z)  

---

## Model
- 1D Convolutional Neural Network (CNN)  
- Activation: ReLU  
- Regularization: BatchNorm + Dropout  
- Loss: Cross-Entropy  
- Optimizer: AdamW  
- Batch size: 64  
- Epochs: 30  

---

## Results

### Dataset Performance
- Accuracy: ~80%  
- Macro-F1: ~0.79  

### Real-World Performance
- Accuracy: ~12%  
- Significant drop due to environmental noise and signal distortion  

---

## Analysis
- Strong performance in controlled datasets  
- Real-world performance degrades due to:
  - Noise and environmental interference  
  - Sensor placement variability  
  - Weak vibration signal propagation  
- Confusion matrix and per-label analysis show class imbalance and misclassification patterns  
- Prediction distribution reveals bias toward dominant classes  

---

## Security Implications
This project demonstrates that motion sensors can act as **unintended side channels**, potentially leaking speech-related information.

This raises important **privacy and security concerns in IoT environments**, where sensors are widely accessible and often lack strict access control.

---

## Limitations
- Significant performance drop in real-world conditions  
- Sensitive to noise and environmental variability  
- Limited real-world dataset size  
- Domain mismatch between training and deployment environments  

---

## Tech Stack
- Python  
- PyTorch  
- NumPy / Pandas  
- Matplotlib / Seaborn  
- Raspberry Pi 5  
- ADXL345 Accelerometer  

---

## Requirements
- Python 3.9+  
- PyTorch  
- NumPy  
- Pandas  
- Matplotlib  
- Seaborn  

---

## Quick Start

```bash
git clone https://github.com/AsamoahAlbert/motion-keyword-inference-imu.git
cd motion-keyword-inference-imu
pip install -r requirements.txt
python src/train_model.py
