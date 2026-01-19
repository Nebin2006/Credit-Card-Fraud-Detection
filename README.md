# 💳 Credit Card Fraud Detection using Random Forest

## 📌 Project Overview
This project focuses on detecting **fraudulent credit card transactions** from a highly **imbalanced dataset**.  
Using a dataset of **~284,807 transactions**, the objective is to accurately identify the **tiny fraction of fraud cases (0.17%)** while minimizing false alarms for legitimate users.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:**  
  - Scikit-learn  
  - Imbalanced-learn (SMOTE)  
  - Pandas  
  - Matplotlib  
  - Seaborn  
- **Environment:** Google Colab  

---

## 🚀 Key Features & Workflow

### 1️⃣ Exploratory Data Analysis (EDA)
- Identified **extreme class imbalance**:
  - **284,315** normal transactions  
  - **492** fraudulent transactions  
- Scaled sensitive features like **`Amount`** and **`Time`** using `StandardScaler` to ensure uniform feature distribution.

---

### 2️⃣ Handling Class Imbalance
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training dataset.
- Balanced fraud cases from **394 to 227,451**, allowing the model to learn fraud patterns without bias.

---

### 3️⃣ Model Development
- Built a **Random Forest Classifier** to capture complex, non-linear patterns in transaction data.
- Used **stratified train-test splitting** to preserve the fraud ratio in both training and testing sets.

---

### 4️⃣ Performance Metrics
Instead of accuracy, the model was evaluated using **fraud-sensitive metrics**:

- **Recall (Fraud Class): 0.84**  
  → Successfully detected **82 out of 98 fraud cases**

- **Precision (Fraud Class): 0.85**  
  → **85%** of predicted frauds were genuine frauds

- **F1-Score: 0.84**  
  → Balanced trade-off between fraud detection and false alarms

---

## 🧠 Critical Thinking & Deployment

### 📉 Business Implication
- A **high recall (84%)** significantly reduces financial losses caused by **missed frauds (False Negatives)**.
- This makes the model suitable for real-world financial systems where fraud detection is critical.

### 🚀 Real-world Deployment
- The trained model is saved as a **`.pkl` file** for fast loading.
- In a production environment (e.g., a bank), this model would be deployed as an **API** to validate transactions in real time.

---

## 📂 Repository Structure
- Fraud_Detection.ipynb # Complete EDA, training, and evaluation
- fraud_model.pkl # Pre-trained Random Forest model

