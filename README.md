# Parkinson-Disease-Prediction-using-Machine-Learning-
🧠 Parkinson’s Disease Prediction using Machine Learning   Parkinson’s Disease (PD) is a progressive neurological disorder that affects movement. In this project, we build a Machine Learning pipeline to predict Parkinson’s disease using biomarkers. 
We cover everything from data preprocessing → feature selection → class balancing → model training → evaluation.

📂 Project Workflow

1️⃣ Import Libraries & Dataset
2️⃣ Data Exploration & Cleaning
3️⃣ Data Wrangling (aggregation by patient)
4️⃣ Remove Multicollinearity
5️⃣ Feature Selection (Chi-Square)
6️⃣ Handle Class Imbalance (Oversampling)
7️⃣ Model Training (Logistic Regression, XGBoost, SVM)
8️⃣ Model Evaluation (ROC AUC, Confusion Matrix, Report)

📊 Dataset Overview

Total Patients: 252

Features: 30 selected biomarkers

Target: class → (0 = Healthy, 1 = Parkinson’s)

Pie chart of class distribution before balancing:

x = df['class'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()

🧹 Data Preprocessing

✔ Aggregated patient records (groupby id → mean)
✔ Removed highly correlated features (>0.7)
✔ Scaled features using MinMaxScaler
✔ Selected Top 30 Features using SelectKBest(chi2)
✔ Balanced dataset using RandomOverSampler

🤖 Models Used

Logistic Regression (Best performer ✅)

XGBoost Classifier (Overfitted ❌)

Support Vector Classifier (Low Validation Accuracy ❌)

🏆 Model Performance
Model	Train ROC AUC	Validation ROC AUC
Logistic Regression	0.75	0.82 ✅
XGBoost Classifier	1.00 (overfit)	0.64
Support Vector Classifier	0.62	0.65
📌 Confusion Matrix (Logistic Regression)
TP = 35 | TN = 10  
FP = 4  | FN = 2


Confusion Matrix visualization:

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(models[0], X_val, Y_val)
plt.show()

📑 Classification Report (Logistic Regression)
              precision    recall  f1-score   support
0.0 (Healthy)    0.77      0.71      0.74        14
1.0 (PD)        0.89      0.92      0.91        37
-----------------------------------------------
Accuracy = 0.86 (86%)

🔎 Key Insights

Logistic Regression gave the best balance of precision (89%) and recall (92%).

XGBoost overfitted due to limited dataset.

SVC underperformed compared to Logistic Regression.

With feature engineering + hyperparameter tuning, accuracy can be further improved.

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/Parkinsons-Prediction-ML.git
cd Parkinsons-Prediction-ML


Install dependencies:

pip install -r requirements.txt


Run the notebook or Python file.

🛠️ Tech Stack

Python 🐍

Pandas, NumPy (Data Handling)

Matplotlib, Seaborn (Visualization)

Scikit-learn (ML Models + Metrics)

Imbalanced-learn (Class balancing)

XGBoost (Boosting Classifier)

✨ Future Improvements

🔹 Hyperparameter Tuning (GridSearchCV, RandomizedSearchCV)
🔹 Ensemble Learning (Stacking Logistic + XGB + SVM)
🔹 Deep Learning (Neural Networks for feature learning)
🔹 Larger datasets for more generalizable models

📌 Conclusion

✔ Machine learning can support early detection of Parkinson’s disease.
✔ Logistic Regression achieved the best validation accuracy (86%).
✔ The pipeline is modular and can be extended with more advanced models.
