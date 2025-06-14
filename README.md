
## 🧬 Breast Cancer Classification using KNN and Gaussian Naive Bayes

### 🧩 Overview

This project compares two fundamental classification algorithms — **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes (GNB)** — for the early detection of breast cancer. Using a structured dataset of tumor characteristics, the goal is to predict whether a tumor is **malignant** or **benign**.

---

### 💼 Business Understanding

Breast cancer diagnosis is one of the most critical tasks in healthcare. With the right algorithms and feature engineering:

* Doctors can receive early diagnostic support
* False negatives can be minimized
* Data-driven decision-making can aid in better patient outcomes

This project demonstrates how **KNN** (a non-parametric lazy learning model) and **GaussianNB** (a probabilistic generative model) can be applied and compared for this task.

---

### 📊 Data Understanding

* **Dataset**: UCI Breast Cancer Wisconsin Dataset (via Kaggle)
* **Size**: 569 samples × 30 numeric features
* **Target**: `diagnosis` — 1 (Malignant), 0 (Benign)
* **Feature Types**: All continuous — mean, standard error, and worst value for cell nucleus properties like:

  * Radius, Texture, Perimeter, Area, Smoothness, Compactness, etc.
* **Preprocessing**:

  * Dropped irrelevant columns (`id`, `Unnamed: 32`)
  * Mapped labels (M → 1, B → 0)
  * Data visualization using `lmplot` for feature correlation exploration

---

### 🤖 Modeling and Evaluation

* **Models Used**:

  * **K-Nearest Neighbors (KNN)** with `n_neighbors=13`
  * **Gaussian Naive Bayes (GNB)**
* **Data Split**:

  * 80% training, 20% testing using `train_test_split`
* **Evaluation Metrics**:

  * Accuracy
  * Classification Report (Precision, Recall, F1-score)
  * Confusion Matrix
* **Results**:

  * Both models performed well; performance was benchmarked and compared
  * Visualization helped interpret feature influence on target class

---

### 📌 Conclusion

* **KNN** and **GaussianNB** proved effective for binary classification on medical data
* **KNN showed slightly better interpretability via distance-based decision boundary**
* This project demonstrates proficiency in model comparison, medical data handling, and visual analysis

---

