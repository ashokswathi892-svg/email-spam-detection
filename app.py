import streamlit as st
import joblib

st.set_page_config(page_title="Spam Detector")

st.title("📧 Email Spam Detection")

try:
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("Model files missing")
    st.stop()

msg = st.text_area("Enter your email message")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("Message type pannunga")
    else:
        data = vectorizer.transform([msg])
        result = model.predict(data)

        if result[0] == 1:
            st.error("🚫 SPAM")
        else:
            st.success("✅ NOT SPAM")

