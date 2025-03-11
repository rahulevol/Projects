import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # ✅ Corrected import
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    precision_score, recall_score
)

st.title("Binary Classification Web App")
st.sidebar.title("Binary Classification Web App")
st.markdown("Are your mushrooms edible or poisonous?🍄")
st.sidebar.markdown("Are your mushrooms edible or poisonous?🍄")

# ✅ Load Data Function
@st.cache_data
def load_data():
    file_path = "D:/Ml/archive/mushrooms.csv"  # Ensure the correct path
    data = pd.read_csv(file_path)

    # Encode categorical variables
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])

    return data

# ✅ Split Data Function
@st.cache_data
def split(df):
    y = df["class"]  # Ensure correct column name
    x = df.drop(columns=["class"])
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test

# ✅ Function to Plot Metrics
def plot_metrics(metrics_list, model, x_test, y_test):
    y_pred = model.predict(x_test)

    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

# Load and Split Data
df = load_data()
x_train, x_test, y_train, y_test = split(df)
class_names = ['edible', 'poisonous']

# Sidebar - Classifier Selection
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

# ✅ SVM Classifier
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C')
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Classify", key='classify_svm'):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy:", round(accuracy, 2))
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')

        st.write("Precision:", round(precision, 2))
        st.write("Recall:", round(recall, 2))

        plot_metrics(metrics, model, x_test, y_test)

# ✅ Logistic Regression Classifier
elif classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    max_iter = st.sidebar.slider("Maximum Number of Iterations", 100, 500, key='max_iter')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Classify", key='classify_lr'):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy:", round(accuracy, 2))
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')

        st.write("Precision:", round(precision, 2))
        st.write("Recall:", round(recall, 2))

        plot_metrics(metrics, model, x_test, y_test)

# ✅ Random Forest Classifier
elif classifier == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", [True, False], key='bootstrap')

    metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Classify", key='classify_rf'):
        st.subheader("Random Forest Classifier Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy:", round(accuracy, 2))
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')

        st.write("Precision:", round(precision, 2))
        st.write("Recall:", round(recall, 2))

        plot_metrics(metrics, model, x_test, y_test)

# ✅ Show Raw Data
if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushrooms Data Set (Classification)")
    st.write(df)
