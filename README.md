# 🧠 Brain Tumor Classification using CNN

This project implements a **Convolutional Neural Network (CNN)** in **TensorFlow/Keras** to classify brain MRI images into four categories:

- **Glioma Tumor**  
- **Meningioma Tumor**  
- **No Tumor**  
- **Pituitary Tumor**

The model is trained on a labeled dataset of MRI scans and achieves high accuracy in distinguishing between different tumor types.

---

## 🚀 Project Workflow
1. **Dataset Preparation**
   - Images are organized in folders by class (`glioma_tumor`, `meningioma_tumor`, `no_tumor`, `pituitary_tumor`).
   - Images are resized to `150x150` and normalized (`0-1` scaling).
   - Labels are one-hot encoded.

2. **Model Architecture**
   - Multiple `Conv2D` + `MaxPooling2D` layers for feature extraction.
   - `Dropout` layers to prevent overfitting.
   - Fully connected (`Dense`) layers for classification.
   - Output layer with **softmax** activation for multi-class prediction.

3. **Training**
   - Loss: `categorical_crossentropy`  
   - Optimizer: `Adam`  
   - Metric: `Accuracy`  
   - Trained for `25 epochs` with validation split.

4. **Evaluation**
   - Train/validation accuracy and loss are plotted.
   - Final evaluation is done on a held-out **test set**.

5. **Prediction**
   - Given a new MRI image, the model predicts the tumor class.  
   - Example:
     ```python
     import cv2
     import numpy as np

     labels = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

     img = cv2.imread("Testing/glioma_tumor/TrAb001.jpg")
     img = cv2.resize(img, (150, 150))
     img_array = np.array(img).reshape(1, 150, 150, 3) / 255.0

     prediction = model.predict(img_array)
     predicted_class = labels[np.argmax(prediction)]
     print("Predicted Tumor Type:", predicted_class)
     ```

---

## 📊 Results
- Training accuracy and validation accuracy converge well.
- Example plot of accuracy/loss curves:

![Training vs Validation Accuracy](<img width="2321" height="1240" alt="image" src="https://github.com/user-attachments/assets/f7a2c6c4-3a11-4c7c-85e8-844f67ad99ce" />
)  
![Training vs Validation Loss](<img width="2320" height="1291" alt="image" src="https://github.com/user-attachments/assets/322572be-7c6b-48a8-b4f4-65e7cdc7ae48" />
)  



---


