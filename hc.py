import joblib
import streamlit as st

# Load trained model and TF-IDF vectorizer
model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered",
)

st.markdown(
    """
<style>
/* Richer Lilac/Purple Background */
.stApp, [data-testid="stAppViewContainer"], .main {
    background: linear-gradient(135deg, #e3d5f3 0%, #ede4f7 50%, #d8c4ee 100%) !important;
    color: #000000 !important;
}

/* Ensure body text and labels stay pitch-black */
.stApp p, .stApp label, .stApp span, .stApp div, .stApp h3 {
    color: #000000 !important;
}

/* Goldenrod title with strong contrast */
.big-title {
    text-align: center;
    color: #DAA520 !important;
    font-size: 50px !important;
    font-weight: 800 !important;
    margin-bottom: 0px !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
}

/* Clear black subtitle */
.subtitle {
    text-align: center;
    color: #1a1a1a !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    margin-top: 5px !important;
}

/* Fix Textarea styling */
textarea, input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #b39ddb !important;
    border-radius: 8px !important;
}

/* Fix Button background, border, and text visibility */
div.stButton > button {
    background-color: #ffffff !important;
    color: #4a148c !important;
    border: 2px solid #8e24aa !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    background-color: #8e24aa !important;
    color: #ffffff !important;
    border-color: #8e24aa !important;
}

div.stButton > button p {
    color: inherit !important;
}

/* ONLY ST.INFO GETS LILAC */
div[data-testid="stAlertContainer-info"] div[data-testid="stAlert"],
div[aria-label="info"] {
    background-color: #d1c4e9 !important;
    border: 1px solid #9c27b0 !important;
}

div[data-testid="stAlertContainer-info"] svg {
    fill: #4a148c !important;
    color: #4a148c !important;
}

/* RESTORE GREEN FOR ST.SUCCESS */
div[data-testid="stAlertContainer-success"] div[data-testid="stAlert"],
div[aria-label="success"] {
    background-color: #d4edda !important;
    border: 1px solid #28a745 !important;
}

/* RESTORE RED FOR ST.ERROR */
div[data-testid="stAlertContainer-error"] div[data-testid="stAlert"],
div[aria-label="error"] {
    background-color: #f8d7da !important;
    border: 1px solid #dc3545 !important;
}

/* Force dark text inside all alert banners */
.stAlert p, .stAlert li, .stAlert strong {
    color: #000000 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="big-title">🛒 Product Review Sentiment Analysis</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Machine Learning Based Sentiment Classifier</p>',
    unsafe_allow_html=True,
)

st.divider()

review = st.text_area(
    "✍ Enter Product Review",
    height=180,
    placeholder="Example: This phone is amazing. Battery backup is excellent.",
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
            st.success(
                "😊 Congratulations! A Positive Review, You can buy this product."
            )
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
