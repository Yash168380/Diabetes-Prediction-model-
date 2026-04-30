import streamlit as st
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("Diabetes_model.pkl")

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Main Background */
.stApp {
    background: linear-gradient(135deg, #eef2f3, #ffffff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3c72, #2a5298);
    padding: 20px;
}

/* Sidebar Heading */
section[data-testid="stSidebar"] h2 {
    color: white;
    font-weight: bold;
}

/* Sidebar Labels */
section[data-testid="stSidebar"] label {
    color: #e0e6f0 !important;
    font-size: 14px;
}

/* Slider numbers */
section[data-testid="stSidebar"] .stSlider span {
    color: white !important;
}

/* Slider color */
.stSlider > div > div {
    color: #00c6ff !important;
}

/* Input box */
section[data-testid="stSidebar"] input {
    background-color: white !important;
    color: black !important;
    border-radius: 8px;
}

/* Cards */
.card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    transition: 0.3s;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 15px 30px rgba(0,0,0,0.15);
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    border-radius: 10px;
    padding: 10px 25px;
    font-size: 16px;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #005bea, #00aaff);
}

/* Result */
.result-high {
    color: #ff4b5c;
    font-size: 26px;
    font-weight: bold;
}
.result-low {
    color: #00c853;
    font-size: 26px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div style="
    padding: 25px;
    border-radius: 15px;
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.15);
    text-align: center;
">
    <h1 style="color:white; margin-bottom:5px;">
        🩺 Diabetes Prediction System
    </h1>
    <p style="color:white; font-size:18px;">
        AI-powered smart health risk analysis
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 📋 Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 110)
bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("DPF", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- LEFT ----------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Patient Summary")

    st.write(f"**Pregnancies:** {pregnancies}")
    st.write(f"**Glucose:** {glucose}")
    st.write(f"**Blood Pressure:** {bp}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Age:** {age}")

    if st.button("🚀 Predict Now"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

        try:
            prediction = model.predict(data)[0]
            prob = model.predict_proba(data)[0][1]

            st.subheader("🧠 Result")

            if prediction == 1:
                st.markdown(f'<p class="result-high">⚠️ High Risk ({prob:.2%})</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="result-low">✅ Low Risk ({prob:.2%})</p>', unsafe_allow_html=True)

            st.session_state["prob"] = prob

        except:
            st.error("❌ Invalid Input")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT ----------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Risk Meter")

    prob = st.session_state.get("prob", 0)
    st.progress(int(prob * 100))
    st.write(f"### {prob:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Info")

    st.write("""
    This system predicts diabetes risk using Machine Learning.

    ⚠️ Not a medical diagnosis.
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<p class="footer">Made with ❤️ | Professional ML App</p>', unsafe_allow_html=True)