import streamlit as st
import numpy as np
import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2

# Original Dataframe
st.subheader("News Dataframe")
df = pd.read_csv("projectdata.csv")
st.dataframe(df)

# Plotting category in a bar chart

st.subheader("Category Plot")
category_counts = df['news_category'].value_counts()
st.bar_chart(category_counts)

# After Data Cleaning

st.subheader("Cleaned DataFrame")
df = pd.read_csv("new_projectdata.csv")
st.dataframe(df)

#  Training Model

from sklearn.linear_model import LogisticRegression
log_regression = LogisticRegression()
vectorizer = TfidfVectorizer(stop_words = "english")
X = df['cleaned'] # independent
Y = df["news_category"] # dependent
X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30)


# Creating Pipeline
pipeline = Pipeline([('vect', vectorizer),
                    ("chi", SelectKBest(chi2, k=1450)),
                     ("clf", LogisticRegression(random_state = 0))])

# Training the model
model = pipeline.fit(X_train, Y_train)

# 30% of data is being used for testing purpose in this case.

# Accuracy
from sklearn.metrics import accuracy_score
predicted_category = model.predict(X_test)
accuracy = accuracy_score(Y_test, predicted_category)
st.subheader("After training the model using logistic regression, we get the accuracy score as: ")
st.write("Accuracy score = ",accuracy )

# Input and input news classification
st.header("News Classification: ")
news = st.text_area("Enter the news")

def news_classification(txt):
    news_data = {'predicted_category':[txt]}
    news_data_df = pd.DataFrame(news_data)
    predict_news_cat = model.predict(news_data_df['predicted_category'])
    return predict_news_cat[0]
if st.button("Submit"):
    st.subheader("Your input news is: ")
    st.text(news)
    st.subheader("Predicted news category =  " + news_classification(news))
    

