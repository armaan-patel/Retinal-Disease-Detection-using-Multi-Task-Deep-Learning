# 🧠 Retinal Disease Detection using Multi-Task Deep Learning

This project implements a **deep learning model** for automated **retinal disease detection** using fundus images from the [Kaggle Retinal Disease Detection dataset](https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection).  
The model performs **two tasks simultaneously** — classifying **diabetic retinopathy severity** and predicting the **risk of macular edema** — while providing **visual explanations** through **Grad-CAM heatmaps**.

---

## 📘 Abstract

Accurate and interpretable retinal image analysis is essential for early diagnosis of diabetic eye diseases.  
This project builds a **multi-task convolutional neural network (CNN)** using **PyTorch** that jointly predicts *Retinopathy Grade* (multi-class) and *Macular Edema Risk* (binary).  
A pretrained **ResNet-50 backbone** is fine-tuned with two classification heads, trained on labeled fundus images with augmentation and class balancing.  
The inclusion of **Grad-CAM** provides transparent visualization of the regions most responsible for each prediction, ensuring the model’s focus aligns with clinical features such as hemorrhages, exudates, and macular changes.

---

## 🔑 Key Features

- **📚 Dataset:**  
  - Kaggle *Retinal Disease Detection* dataset with annotated fundus images and clinical captions.  
  - Columns: `Image name`, `Retinopathy grade`, `Risk of macular edema`, `Caption`.

- **🧩 Multi-Task Deep Learning Architecture:**  
  - **Shared Backbone:** ResNet-50 pretrained on ImageNet.  
  - **Head 1:** Retinopathy Grade (multi-class classification).  
  - **Head 2:** Macular Edema Risk (binary classification).  
  - Weighted cross-entropy loss for balanced optimization.

- **⚙️ Training Pipeline:**  
  - Data augmentation (resize, flip, rotation, color jitter).  
  - Weighted sampling for class imbalance.  
  - Learning rate scheduling and checkpoint saving.

- **🎯 Evaluation:**  
  - Accuracy, precision, recall, F1-score, and confusion matrices.  
  - Separate metrics for training, validation, and test sets.

- **🧠 Explainability:**  
  - **Grad-CAM** visualizations show lesion regions influencing the predictions.  
  - Highlights peripheral lesions for retinopathy and macular regions for edema.

---

## 🧰 Technical Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.11 |
| **Frameworks** | PyTorch, Torchvision |
| **Data Processing** | Pandas, NumPy, Pillow |
| **Visualization** | Matplotlib, Seaborn, OpenCV |
| **Metrics & Analysis** | scikit-learn |
| **Environment** | JupyterLab / Kaggle Notebook (GPU runtime) |

---

## 🧪 Results

| Task | Metric | Training | Validation | Test |
|:------|:--------|:-----------:|:-------------:|:--------:|
| **Retinopathy Grade** | Accuracy | ~86% | ~69% | ~72% |
| **Macular Edema Risk** | Accuracy | ~99% | ~94% | ~96% |

(Note: Results vary slightly based on random seed, augmentations, and fine-tuning strategy.)

### Example Grad-CAM Outputs

| Original Fundus | DR Head | ME Head |
|:----------------:|:--------:|:--------:|
| <img width="529" height="350" alt="image" src="https://github.com/user-attachments/assets/8c09c6e3-07cc-4860-881b-5eea7a012c2b" /> | <img width="529" height="351" alt="image" src="https://github.com/user-attachments/assets/8970b031-9c3c-4a7d-a4b1-252874db39e1" /> | <img width="530" height="351" alt="image" src="https://github.com/user-attachments/assets/397cb6f7-7dae-4019-8498-3381bf937d7c" /> |

- **DR Head:** Highlights peripheral lesions and hemorrhages associated with diabetic retinopathy.  
- **ME Head:** Concentrates on the central macula region to evaluate edema risk.

---

## 🧩 Discussion

### Strengths
- Multi-task learning enhances shared representation and reduces overfitting.  
- Grad-CAM explainability confirms attention to clinically relevant features.  
- High overall accuracy across both disease classification tasks.

### Limitations
- Occasional confusion between adjacent retinopathy grades (e.g., moderate vs. severe).  
- Dataset size and label variation limit clinical generalization.

### Future Work
- Fine-tune all ResNet-50 layers at a reduced learning rate.  
- Increase input resolution for finer lesion detection.  
- Add multimodal training using the *Caption* column.  
- Export the trained model via ONNX/TorchScript for deployment.

---

## 📜 Acknowledgments

- **Dataset:** [Kaggle — Retinal Disease Detection](https://www.kaggle.com/datasets/mohamedabdalkader/retinal-disease-detection) by Mohamed Abdalkader.  
- Built using **PyTorch** with a pretrained **ResNet-50** backbone and standard medical imaging best practices.  
- Inspired by current research in **Explainable AI (XAI)** for ophthalmology.

---

## 🧾 License

This project is released under the **MIT License**.  
You are free to use, modify, and share for educational or research purposes with proper credit.

---

## 👤 Author

**Armaan Patel**  
B.S. Computer Science, DeSales University

