# Heart-Disease-Prediction-Model
Heart Disease Prediction Using Machine Learning
A machine learning project that predicts the likelihood of heart disease based on patient medical data. This project demonstrates data preprocessing, feature engineering, model training, evaluation, and deployment readiness using a healthcare dataset.

📌 Objective
To build and evaluate machine learning models that estimate the risk of heart disease using patient health indicators like age, sex, cholesterol level, chest pain type, and more.

🧠 Dataset Description
The dataset contains the following features:

age: Age of the patient

sex: Gender (1 = male, 0 = female)

cp: Chest pain type (0–3)

trestbps: Resting blood pressure

chol: Serum cholesterol (mg/dl)

fbs: Fasting blood sugar (> 120 mg/dl)

restecg: Resting electrocardiographic results

thalach: Maximum heart rate achieved

exang: Exercise-induced angina

oldpeak: ST depression induced by exercise

slope: Slope of peak exercise ST segment

ca: Number of major vessels (0–3)

thal: Thalassemia type

target: Heart disease presence (1 = yes, 0 = no)

🛠️ Technologies Used

Python 3

Jupyter Notebook

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Joblib for saving model

VS Code as IDE

📊 Workflow

1. Data Preprocessing

Handled missing values (if any)

Standardized numerical columns

Encoded categorical features using One-Hot Encoding

2. Exploratory Data Analysis

Countplots and heatmaps to understand feature relationships

Correlation analysis to identify key predictors

3. Feature Engineering

Created interaction features (e.g., cp_exang, age_oldpeak)

Grouped features into bins (age_group, chol_level)

4. Model Training
   
Trained the following models:

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

5. Evaluation Metrics
   
Accuracy, Precision, Recall, F1 Score

ROC-AUC Score

ROC Curve visualization

6. Model Selection
   
Random Forest was selected based on highest accuracy and AUC score

📁 Project Structure

heart-disease-prediction/

├── heart.csv # Dataset

├── main.ipynb # Jupyter notebook (complete training & evaluation)

├── rf_model.pkl # Trained Random Forest model

├── scaler.pkl # StandardScaler used during training

├── columns.pkl # Feature columns used for input transformation
