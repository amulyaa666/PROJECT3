# PCA with ANN for Face Recognition

## 📌 Project Overview
This project implements a **Face Recognition System** using **Principal Component Analysis (PCA)** for feature extraction and **Artificial Neural Network (ANN)** for classification.  
PCA reduces high-dimensional facial image data into eigenfaces, and ANN classifies individuals based on these features.

This project was developed as part of the **Internship Studio Program**.

---

## 🧠 Technologies Used
- Python 3.11
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

---

## 📂 Project Structure


---

## 📊 Dataset Description
- The dataset consists of **grayscale facial images**
- Images are organized **person-wise**
- Each image is resized to **100 × 100 pixels**
- Dataset is split into:
  - **60% Training**
  - **40% Testing**

---

## ⚙️ Methodology
1. Image acquisition
2. Image preprocessing
3. Mean face calculation
4. PCA feature extraction (Eigenfaces)
5. ANN training using backpropagation
6. Testing and evaluation

---

## 📈 Results
- Best accuracy achieved at **k = 30 eigenfaces**
- Accuracy decreases for higher k due to overfitting
- PCA + ANN provides efficient dimensionality reduction and classification

---

## 🚀 How to Run the Project

### Step 1: Install dependencies
```bash
pip install -r requirements.txt

```
### Step 2: Run the program
```bash
python main.py

```

---

✅ Advantages

Reduced dimensionality

Faster computation

Improved recognition accuracy

---


⚠️ Limitations

Sensitive to lighting variations

Performance depends on dataset size

---

🔮 Future Scope

Use CNN for higher accuracy

Real-time face recognition

Larger datasets

---

📄 Documentation

Full project documentation is available in the documentation folder.
