# 🏍️ Boda Boda Safety Object Detection

This project uses **YOLOv8**, a state-of-the-art object detection model, to identify key safety compliance features among boda boda (motorcycle taxi) riders — such as **helmets**, **reflectors**, and **overloading**. The goal is to contribute to road safety initiatives in Kenya by providing a computer vision-based approach to monitoring safety adherence.

## 📂 Project Overview
The model was trained using a custom dataset collected and labeled through **Roboflow**, consisting of images categorized into:
- **helmet**
- **no_helmet**
- **reflector**
- **no_reflector**
- **overload**
- **no_overload**

Training and validation were performed in **Google Colab** using the **Ultralytics YOLOv8 framework**.

## 🧠 Objectives
- Detect whether boda boda riders are wearing helmets.
- Identify the presence of reflective clothing.
- Detect overloading cases (more than one passenger or excess cargo).

## ⚙️ Model Training
- **Base model:** `yolov8s.pt`
- **Training epochs:** 60
- **Image size:** 640x640
- **Batch size:** 16  
- **Tools:** Roboflow for dataset preparation, Ultralytics YOLO for training and evaluation.

## 📊 Evaluation Metrics
The trained model achieved:
- **Precision:** ~0.85  
- **Recall:** ~0.81  
- **mAP@50:** ~0.85  
- **mAP@50-95:** ~0.56  

These results show good detection performance, especially for helmets and reflectors, with room for improvement in the *no_helmet* and *no_reflector* classes.


