# 🛞 Tire Defect Detection using YOLOv8

A real-time deep learning project to detect and classify tire defects such as bulges, cracks, and flat spots using the YOLOv8 object detection model.

---

## 🔍 Objective

The goal of this project is to overcome the limitations of traditional sensor-based tire defect detection systems (like in the research paper) by using a camera-based, AI-powered solution that:
- Works in real-time
- Requires no specialized hardware
- Supports multiple defect types

---

## 🚀 Features

- Detects 4 classes: **Bulge, Cracks, Flat Spots, Non-defective**
- Trained using **YOLOv8n** (Ultralytics)
- Works with **static images** and can be extended to video/webcam
- Real-time feedback with bounding boxes
- Easy deployment and portable

---

## 📂 Dataset

- Labeled dataset from Roboflow in YOLO format
- Classes: `['Bulge', 'Cracks', 'Flat spots', 'Non-defective']`

This model is also deployed in Huggingface
link: https://huggingface.co/spaces/Foxy-Roxy/Wheel_Defect_Detection
Link: https://defective-tyre-detection.onrender.com/
