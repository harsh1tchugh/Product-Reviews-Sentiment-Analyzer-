import streamlit as st
import joblib

# Load trained model and TF-IDF vectorizer
model = joblib.load("C:/Users/Harshit/Downloads/best_sentiment_model.pkl")
tfidf = joblib.load("C:/Users/Harshit/Downloads/tfidf_vectorizer.pkl")
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

st.markdown("""
<style>
.main{
    background-color:#f7f9fc;
}
.big-title{
    text-align:center;
    color:#1f77b4;
    font-size:40px;
    font-weight:bold;
}
.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}
.result{
    font-size:25px;
    font-weight:bold;
    text-align:center;
    padding:15px;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🛒 Amazon Review Sentiment Analysis</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Machine Learning Based Sentiment Classifier</p>', unsafe_allow_html=True)

st.divider()

review = st.text_area(
    "✍ Enter Amazon Review",
    height=180,
    placeholder="Example: This phone is amazing. Battery backup is excellent."
)

col1, col2 = st.columns(2)

with col1:
    predict = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear = st.button("🗑 Clear", use_container_width=True)

if predict:

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        review_vector = tfidf.transform([review])

        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
            st.balloons()

        else:
            st.error("😞 Negative Review")

st.divider()

st.markdown("### 📊 About")

st.info("""
**Dataset:** Amazon Review Polarity Dataset

**Algorithms Used**
- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- Linear SVM
- KNN

**Feature Extraction**
- TF-IDF Vectorization

**Developed using**
- Python
- Scikit-learn
- Streamlit
""")


