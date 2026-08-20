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
/* Purplish-whitish soft gradient background */
.main {
    background: linear-gradient(135deg, #f3f0f7 0%, #ffffff 50%, #eae4f2 100%);
    color: #000000;
}

/* Title with Goldenrod color and subtle text glow */
.big-title {
    text-align: center;
    color: #DAA520; /* Goldenrod */
    font-size: 54px;
    font-weight: bold;
    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.15);
}

/* Subtitle with dark black text */
.subtitle {
    text-align: center;
    color: #1a1a1a;
    font-size: 18px;
    font-weight: 500;
}

/* Result box with light purple fill, border, and black text */
.result {
    font-size: 25px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 10px;
    background-color: #f0eaf8;
    color: #000000;
    border: 1px solid #dcd0ea;
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

