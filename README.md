# AI-Based Disease Detection System

A comprehensive Deep Learning system designed to detect and classify multiple types of diseases using medical imaging and data. This project utilizes pre-trained models and custom training scripts to identify conditions related to the **Lung, Brain, Heart, and Skin**.

## 🚀 Features

This system integrates four distinct disease detection models:

* **🫁 Lung Cancer Detection:** Analyzes scans to identify potential malignancies (Large model handled via Git LFS).
* **🧠 Brain Tumor Detection:** Classifies brain MRI scans (`processed_brainblock`).
* **❤️ Heart Disease Detection:** Analyzes heart-related data/ECG (`processed_heart`, `heartbeat`).
* **wm Skin Disease Detection:** Identifies skin conditions from dermatological images (`processed_skin_final`).

## 📂 Project Structure

```text
AI-Based-Disease-Detection-System/
├── 📁 LungcancerDataSet/       # Data for lung cancer training
├── 📁 processed_brainblock/    # Pre-processed brain scan data
├── 📁 processed_heart/         # Pre-processed heart data
├── 📁 processed_skin_final/    # Pre-processed skin disease images
├── 📁 heartbeat/               # Raw heartbeat/ECG data
│
├── 🧠 Models (H5 & TFLite):
│   ├── lung_cancer_model.h5    # (Note: Large file, requires Git LFS)
│   ├── brainblock_model.h5
│   ├── heart_model.h5
│   ├── skin_disease_model.h5
│   └── *.tflite                # Lightweight versions for mobile/edge deployment
│
├── 📜 Scripts:
│   ├── preprocess_all.py       # Script to preprocess raw datasets
│   ├── train_all_models.py     # Script to retrain all models
│   └── python.py               # Utility script
