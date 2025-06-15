
## 🧬 Breast Cancer Classification using KNN and Gaussian Naive Bayes

### 🧩 Overview
This project addresses the problem of **classifying breast cancer tumors** as either **malignant** or **benign** by using the **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes** algorithms.The analysis uses a dataset from the UCI Machine Learning Repository, provided by Kaggle, which contains various features of breast cancer patients.

---

### 💼 Business Understanding

The **World Health Organization (WHO)** reports that breast cancer accounts for a significant portion of cancer deaths in women, highlighting the severity of malignant breast cancer. **In 2022, 670,000 women died from breast cancer globally**. The problem is to accurately predict whether a breast tumor is malignant or benign, which can aid in early detection and treatment planning.The core problem aligns with critical needs in healthcare for reliable diagnostic tools. The primary stakeholder is likely medical professionals or organizations involved in cancer diagnosis.When discussing breast cancer deaths, the focus is primarily on malignant tumors because these are the ones that pose a significant risk to life. With such models for early detection of the tumors, 

* Doctors can receive early diagnostic support
* False negatives can be minimized
* Data-driven decision-making can aid in better patient outcomes

This project demonstrates how **KNN** (a non-parametric lazy learning model) and **GaussianNB** (a probabilistic generative model) can be applied and compared for this task.

---

### 📊 Data Understanding
The data used in this project is a dataset of breast cancer patients.The dataset contains 569 entries and 33 columns initially.The data has a class imbalance, with a higher number of benign cases compared to malignant cases, as shown by the value counts and pie chart of the 'diagnosis' column. Visualizations like the lmplot helped understand the relationship between features and the diagnosis.

1. DATA SUMMARY

* **Dataset**: UCI Breast Cancer Wisconsin Dataset (via Kaggle)
* **Size**: 569 samples × 30 numeric features
* **Target**: `diagnosis` — 1 (Malignant), 0 (Benign)
* **Feature Types**: All continuous — mean, standard error, and worst value for cell nucleus properties like:

     ![image](https://github.com/user-attachments/assets/5864b1fe-4829-4f2b-83e4-fc6487f8c635)

  * Radius, Texture, Perimeter, Area, Smoothness, Compactness, etc.
* **Preprocessing**:

  * Dropped irrelevant columns (`id`, `Unnamed: 32`)
  * Mapped labels (M → 1, B → 0)
  * Data visualization using `lmplot` for feature correlation exploration
 
![image](https://github.com/user-attachments/assets/86f241b3-e587-4c07-8dd8-d04f5a932e21)
![image](https://github.com/user-attachments/assets/b94816a7-827e-4ca9-9377-52a3b26a3e79)





---

### 🤖 Modeling and Evaluation

* **Models Used**:Two classification models were used: K-Nearest Neighbors (KNN) and Gaussian Naive Bayes.
  * **K-Nearest Neighbors (KNN)** with `n_neighbors=13`:- A KNN model was built and evaluated. The project explored different values for the number of neighbors (k) using cross-validation with 10 folds to find the optimal k based on minimizing the misclassification error. The model's accuracy was then reported.
  * **Gaussian Naive Bayes (GNB)** A Gaussian Naive Bayes model was also built and trained on the data. The model's accuracy was reported, and predictions on the first 10 test data points were shown.
The primary evaluation metric used for both models is accuracy. Cross-validation was used with the KNN model to find the optimal hyperparameters.

* **Data Split**:

  * 80% training, 20% testing using `train_test_split`
* **Evaluation Metrics**:

  * Accuracy
  * Classification Report (Precision, Recall, F1-score)
    
* **Results**:

  * Both models performed well; performance was benchmarked and compared
  * Visualization helped interpret feature influence on target class

---

### 📌 Conclusion

In this project, both **K-Nearest Neighbors (KNN)** and **Gaussian Naive Bayes (GaussianNB)** models demonstrated strong performance in classifying breast tumors as benign or malignant. The KNN model, specifically when trained on the original data, **achieved high accuracy (97.4%) with perfect precision (1.00) and high recall (0.93) for identifying malignant cases, effectively minimizing false positives and negatives in this critical task. Similarly, the GaussianNB model, even with SMOTE balancing, showed comparable high accuracy and excellent performance metrics for malignant tumor identification.** Both models are promising tools for breast cancer diagnosis. Future work could involve exploring other models, further hyperparameter tuning, and potentially acquiring more data to enhance performance.

This project demonstrates proficiency in model comparison, medical data handling, and visual analysis

---

