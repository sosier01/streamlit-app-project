# your code here
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.title("Iris Flower Prediction App")

# Load data
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

st.header("Input Flower Measurements")

# Sliders for user input
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.4)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.4)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.3)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Prediction
prediction = model.predict(input_data)[0]
probabilities = model.predict_proba(input_data)[0]

st.subheader("Prediction")
st.write(f"🌼 Predicted species: **{target_names[prediction]}**")

# Show probabilities
st.subheader("Prediction Probabilities")

fig, ax = plt.subplots()
ax.bar(target_names, probabilities)
ax.set_ylabel("Probability")
st.pyplot(fig)