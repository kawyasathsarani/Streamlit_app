# Streamlit_app

# 🚢 Titanic Survival Prediction Streamlit Web App

A Machine Learning-powered interactive web application built using **Streamlit** that predicts whether a passenger survived the Titanic disaster based on personal and travel information.

This project demonstrates the complete ML workflow including:

- Data preprocessing  
- Model training  
- Model saving  
- Web app deployment using Streamlit  

---

---

## 📌 Table of Contents

- Project Overview  
- Problem Statement  
- Dataset Information  
- Features of the App  
- Technologies Used  
- Project Structure  
- Model Training  
- How to Run the App  
- App Usage  
- Screenshots  
- Deployment  
- Future Improvements  
- License  
- Author  

---

---

## 📖 Project Overview

The sinking of the Titanic is one of the most famous tragedies in history.  
This project uses Machine Learning algorithms to predict the survival chances of Titanic passengers.

The Streamlit app allows users to enter passenger details such as:

- Age  
- Gender  
- Passenger Class  
- Fare  
- Embarkation Port  

and instantly predicts whether the passenger would survive.

---

---

## ❓ Problem Statement

Given passenger details, predict:

✅ **Survived**  
or  
❌ **Did Not Survive**

This is a **binary classification problem**.

---

---

## 📊 Dataset Information

The dataset used in this project is the famous **Titanic Dataset** from Kaggle.

### Dataset Features:

| Column Name   | Description |
|-------------|-------------|
| Pclass      | Passenger class (1st, 2nd, 3rd) |
| Sex         | Gender of passenger |
| Age         | Age in years |
| SibSp       | # of siblings/spouses aboard |
| Parch       | # of parents/children aboard |
| Fare        | Passenger fare |
| Embarked    | Port of embarkation |

### Target Column:

| Column | Meaning |
|--------|---------|
| Survived | 1 = Survived, 0 = Not Survived |

---

---

## 🎯 Features of the Application

✅ Simple and interactive UI  
✅ Real-time prediction output  
✅ Machine Learning model integration  
✅ User input through dropdowns and sliders  
✅ Beginner-friendly Titanic ML project  
✅ Deployable on Streamlit Cloud  

---

---

## 🛠️ Technologies Used

| Tool/Library | Purpose |
|-------------|---------|
| Python      | Programming language |
| Streamlit   | Web application framework |
| Pandas      | Data handling |
| NumPy       | Numerical operations |
| Scikit-learn| ML model training |
| Pickle/Joblib | Saving trained model |
| Matplotlib/Seaborn | Data visualization (optional) |

---

---

## 📂 Project Structure

titanic-streamlit-app/
│
├── data/
│ └── Titanic-Dataset.csv # Dataset file
│
├── models/
│ └── best_model.joblib # Saved best ML model
│
├── notebooks/
│ └── model_training.ipynb # Training notebook
│
├── app.py # Main Streamlit application
├── model.pkl # Pickle model file (optional)
├── requirements.txt # Required libraries
├── README.md # Project documentation


---

---

## 🤖 Model Training

Model training was performed inside:


### Steps Followed:

1. Data Cleaning  
2. Handling missing values  
3. Encoding categorical variables  
4. Splitting into training/testing sets  
5. Training multiple ML models  
6. Selecting the best-performing model  
7. Saving the best model using Joblib  

### Saved Model Location:


Example saving code:

```python
import joblib
joblib.dump(model, "models/best_model.joblib")


🚀 How to Run the Streamlit App Locally

1️⃣ Clone the Repository

git clone https://github.com/your-username/titanic-streamlit-app.git
cd titanic-streamlit-app

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Streamlit App

streamlit run app.py

4️⃣ Open in Browser

Streamlit will generate a local URL such as:

http://localhost:8501


🎮 App Usage Guide

Enter passenger details:

Passenger Class

Gender

Age

Fare

Family members

Click the Predict Survival button

The app will display the result:

✅ Passenger Survived
or
❌ Passenger Did Not Survive



📸 Screenshot Preview

To add your Streamlit app screenshot:

Step 1: Create folder
images/

Step 2: Save screenshot inside:
images\Screenshot 2026-02-17 210009.png
images\Screenshot 2026-02-17 210035.png
images\Screenshot 2026-02-17 210106.png
images\Screenshot 2026-02-17 210158.png
images\Screenshot 2026-02-17 210339.png
images\Screenshot 2026-02-17 210400.png
images\Screenshot 2026-02-17 210430.png
images\Screenshot 2026-02-17 210440.png
images\Screenshot 2026-02-17 210501.png


Step 3: Display in README


🌍 Deployment

This app can be deployed easily using:

Streamlit Cloud

Render

Heroku

Streamlit Cloud Steps:

Push project to GitHub

1.Go to https://streamlit.io/cloud

2.Click New App

3.Select repository

4.Choose app.py

5.Deploy 🎉

🔮 Future Improvements

🚀 Add more advanced models (XGBoost, Gradient Boosting)
🚀 Improve UI design with custom CSS
🚀 Display prediction probability
🚀 Add data visualization dashboard
🚀 Deploy with a public link

👩‍💻 Author

Developed by: Kawya Sathsarani
🎓 IT Undergraduate 
📍 Horizoncampus

GitHub: https://github.com/kawyasathsarani/Streamlit_app.git