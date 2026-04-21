import streamlit as st
import pickle
import os

# 🔥 FIRST: ensure model exists
if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
    import text_model  # this will create them

# THEN load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

st.title("📧 Spam Message Classifier")
st.write("Enter a message to check whether it is Spam or Not Spam")

msg = st.text_area("Message")

if st.button("Check"):
    if msg.strip() == "":
        st.warning("⚠ Please enter a message")
    else:
        vec = vectorizer.transform([msg])
        result = model.predict(vec)
        prob = model.predict_proba(vec)

        if result[0] == 1:
            st.error(f"🚫 Spam Message (Confidence: {prob[0][1]*100:.2f}%)")
        else:
            st.success(f"✅ Not Spam (Confidence: {prob[0][0]*100:.2f}%)")
