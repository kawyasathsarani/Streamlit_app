# ------------------------------
# Titanic Survival Prediction App
# ------------------------------

import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie
import requests

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction App",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join("data", "Titanic-Dataset.csv")
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset not found at {data_path}")
        st.stop()
    df = pd.read_csv(data_path)
    
    # Convert numeric columns
    num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Feature engineering
    if 'FamilySize' not in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    if 'IsAlone' not in df.columns:
        df['IsAlone'] = df['FamilySize'].apply(lambda x: 1 if x==1 else 0)
    return df

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join("model.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Model not found at {model_path}")
        st.stop()
    with open(model_path,"rb") as file:
        return pickle.load(file)

df = load_data()
model = load_model()

# ------------------------------
# Load Lottie Animation
# ------------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_ship = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_x62chJ.json")

# ------------------------------
# Sidebar Navigation
# ------------------------------
st.sidebar.markdown('<h2 style="color:#ffffff;">📚 Navigation</h2>', unsafe_allow_html=True)
menu = st.sidebar.radio("", ["🏠 Home","📊 Data Exploration","📈 Visualizations","🔮 Prediction","📋 Model Performance"])

# ------------------------------
# Home Page
# ------------------------------
if menu == "🏠 Home":
    st.markdown('<h1 style="text-align:center;color:#2E86C1;">🚢 Titanic Survival Prediction App</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center;color:#555;">Welcome to the Titanic Explorer! Analyze survival patterns and predict passenger survival using AI.</p>',
        unsafe_allow_html=True
    )
    
    # Lottie Animation
    if lottie_ship:
        st_lottie(lottie_ship, height=200, key="ship")
    
    # Titanic image
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg",
        use_container_width=True,
        caption="RMS Titanic - The Unsinkable Ship"
    )
    
    st.markdown("---")
    st.header("Quick Stats Overview")
    col1, col2, col3 = st.columns(3)
    survived = df['Survived'].sum()
    total = len(df)
    died = total - survived

    with col1:
        st.metric("Total Survivors", survived, f"{survived/total:.1%}")
    with col2:
        st.metric("Total Passengers", total)
    with col3:
        st.metric("Total Deceased", died, f"{died/total:.1%}")

# ------------------------------
# Data Exploration
# ------------------------------
elif menu == "📊 Data Exploration":
    st.title("Data Exploration")
    st.dataframe(df.head(10))
    st.write(df.describe())
    
    st.subheader("Survival Rate Pie Chart")
    fig, ax = plt.subplots()
    df['Survived'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', labels=['Died','Survived'], 
        colors=['#e63946','#2a9d8f'], startangle=90, explode=[0.05,0.05], shadow=True, ax=ax
    )
    ax.set_ylabel('')
    st.pyplot(fig)

# ------------------------------
# Visualizations
# ------------------------------
elif menu == "📈 Visualizations":
    st.title("Visualizations")
    
    st.subheader("Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("Survival by Gender")
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1', ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# ------------------------------
# Prediction
# ------------------------------
elif menu == "🔮 Prediction":
    st.title("Predict Survival")
    pclass = st.selectbox("Passenger Class", [1,2,3])
    sex = st.selectbox("Sex", ["male","female"])
    sex = 1 if sex=="male" else 0
    age = st.slider("Age",0,80,25)
    fare = st.number_input("Fare",0.0,500.0,32.0)
    embarked = st.selectbox("Embarked", ["C","Q","S"])
    embarked_map = {"C":0,"Q":1,"S":2}
    embarked = embarked_map[embarked]
    family_size = st.slider("Family Size",1,10,1)
    is_alone = 1 if family_size==1 else 0

    if st.button("Predict"):
        try:
            features = np.array([[pclass,sex,age,fare,embarked,family_size,is_alone]])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0][pred] if hasattr(model,"predict_proba") else None
            if pred==1:
                st.success(f"✅ Passenger would survive! Confidence: {proba:.2f}" if proba else "✅ Passenger would survive!")
            else:
                st.error(f"❌ Passenger would not survive. Confidence: {proba:.2f}" if proba else "❌ Passenger would not survive.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------------
# Model Performance
# ------------------------------
elif menu == "📋 Model Performance":
    st.title("Model Performance")
    df_enc = df.copy()
    df_enc['Sex'] = df_enc['Sex'].map({'male':1,'female':0})
    if 'Embarked' in df_enc.columns:
        df_enc['Embarked'] = df_enc['Embarked'].map({"C":0,"Q":1,"S":2})
    X = df_enc[['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone']]
    y = df_enc['Survived']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    st.write(f"**Accuracy:** {acc:.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test,y_pred))

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown('<div style="text-align:center;color:#6c757d;padding:1rem;">Created by Kawya ❤</div>', unsafe_allow_html=True)
