import streamlit as st
import joblib 

# Load trained model and TF-IDF vectorizer
model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")  # Changed variable name to 'tfidf'

st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)

st.markdown(
    """
<style>
/* Force the root app container to purplish-white */
.stApp, [data-testid="stAppViewContainer"], .main {
    background: #f4effa !important; /* Soft lavender-white */
    color: #000000 !important;
}

/* Ensure all global text, labels, and paragraph elements are black */
.stApp p, .stApp label, .stApp span, .stApp div {
    color: #000000 !important;
}

/* Goldenrod title */
.big-title {
    text-align: center;
    color: #DAA520 !important;
    font-size: 54px;
    font-weight: bold;
    margin-bottom: 5px;
}

/* Clear black subtitle */
.subtitle {
    text-align: center;
    color: #000000 !important;
    font-size: 18px;
    font-weight: 500;
}

/* Custom Result Box with clear dark text */
.result {
    font-size: 22px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    background-color: #e6dbf4 !important;
    color: #000000 !important;
    border: 1px solid #c8b3e6 !important;
}

/* Fix input text box background and text readability */
textarea, input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #dcd0ea !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">🛒 Product Review Sentiment Analysis</p>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Machine Learning Based Sentiment Classifier</p>', unsafe_allow_html=True)

st.divider()

review = st.text_area(
    "✍ Enter Product Review",
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
            st.success("😊 Congratulations! A Positive Review,You can buy this product.")
            st.balloons()

        else:
            st.error("😞 Oops...!!! Negative Review")

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

