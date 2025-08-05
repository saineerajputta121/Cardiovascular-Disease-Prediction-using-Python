# 🫀 Cardiovascular Disease Prediction (Python Project)

## 📌 Project Overview

In this project I had  analyzed  cardiovascular disease data using Python. The main goal is to clean the data, explore relationships between variables,and use a machine learning model to predict whether a person is likely to have cardiovascular disease (CVD).

---

## 🧾 Objective

- Load the dataset into Python
- Perform data cleaning and exploratory data analysis (EDA)
- Understand the relationship between categorical and numerical variables with the target (CVD)
- Use correlation analysis to find important features
- Apply logistic regression to predict CVD
- Evaluate the model accuracy

---

## 🛠️ Tools & Libraries Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📂 Dataset Information

The dataset contains the following columns:

- `age`
- `sex`
- `cp` – chest pain type
- `trestbps` – resting blood pressure
- `chol` – cholesterol level
- `fbs` – fasting blood sugar
- `restecg` – resting electrocardiographic results
- `thalach` – maximum heart rate achieved
- `exang` – exercise-induced angina
- `oldpeak` – ST depression
- `slope` – slope of the peak exercise ST segment
- `ca` – number of major vessels colored by fluoroscopy
- `thal` – thalassemia
- `target` – 1 = CVD present, 0 = no CVD

## ✅ Tasks Performed (with Python Code)

### 1. Import Required Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
```
### 2. Load Dataset and Preview
```python
data = pd.read_excel("simplilearn_medical_datset.xlsx")
data.head()
```
### 3. Preliminary Analysis
```python
print(data.shape)
print(data.info())
print(data.isnull().sum())
```
### 4. Remove Duplicates
```python
duplicate = data[data.duplicated()]
print(duplicate)
data.drop_duplicates(inplace=True)
```
### 5. Statistical Summary
```python
print(data.describe())
```
### 6.  Exploratory Data Analysis (EDA)
  ### 📊 Count Plots for Categorical Columns
```python
categorical_cols = ['target', 'thal', 'cp', 'sex', 'fbs', 'restecg', 'exang']

for col in categorical_cols:
    sns.countplot(x=data[col])
    plt.title(f'{col} Distribution')
    plt.show()
```
 ###🎂 Age Distribution
```python
sns.histplot(x=data['age'], kde=True)
plt.title("Age Distribution - All Patients")
plt.show()
```
###Patients with Cardiovascular Disease
```python
data_cvd = data[data['target'] == 1]
sns.histplot(x=data_cvd['age'], kde=True)
plt.title("Age Distribution - CVD Patients")
plt.show()
```
###👥 Gender Distribution vs CVD
```python
sns.countplot(x=data['sex'], hue=data['target'])
plt.title('Gender vs CVD')
plt.show()
```
###💓 Resting Blood Pressure (CVD Patients Only)
```python
sns.histplot(x=data_cvd['trestbps'], kde=True)
plt.title("Resting BP - CVD Patients")
plt.show()
```
###🧪 Cholesterol Levels (CVD Patients Only)
```python
sns.histplot(x=data_cvd['chol'], kde=True)
plt.title("Cholesterol - CVD Patients")
plt.show()
```
###🏃‍♂️ Exercise-Induced Angina vs CVD
```python
sns.countplot(x=data['exang'], hue=data['target'], color='b')
plt.title("Exercise-Induced Angina vs CVD")
plt.show()
```
###🧬 Thalassemia Types vs CVD
```python
sns.countplot(x=data['thal'], hue=data['target'], color='b')
plt.title("Thalassemia Types vs CVD")
plt.show()
```
###🔗 Correlation Heatmap
```python
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```
### .7 Train-Test Split and Preprocessing
```python
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
 Logistic Regression Model and Accuracy
```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
```

## ✅ Final Conclusion

In this project, I explored and analyzed a cardiovascular disease dataset using Python. After cleaning the data and performing exploratory data analysis (EDA), I was able to identify important patterns and relationships:

- I found that **Type-2 Thalassemia** is a significant factor in the presence of cardiovascular disease.
- Features like **chest pain type (cp)**, **maximum heart rate (thalach)**, and **ST segment slope** showed a **positive correlation** with the target (CVD).
- Other features such as **cholesterol**, **resting blood pressure**, and **fasting blood sugar** showed **negative or weak correlation**.
- By visualizing exercise-induced angina (`exang`), I observed that people with more regular exercise had lower chances of CVD.
- I also saw that **age** and **gender** play an important role, with older individuals and males showing more CVD cases.

Finally, I built a **logistic regression model** to predict the presence of cardiovascular disease. The model achieved an accuracy of **82%**, which is a strong result for a first machine learning approach.

This project helped me improve my skills in data preprocessing, visualization, correlation analysis, and machine learning using Python.













