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

# ---------------- CLEAN MODERN CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(120deg, #fdfbfb, #ebedee);
}

/* Header */
.header {
    padding: 20px;
    border-radius: 15px;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    text-align: center;
    color: white;
    font-weight: bold;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.15);
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
    margin-bottom: 15px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #f8f9fc;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 8px;
    padding: 8px 20px;
    border: none;
}
.stButton>button:hover {
    opacity: 0.9;
}

/* Result */
.result-high {
    color: #e74c3c;
    font-size: 22px;
    font-weight: bold;
}
.result-low {
    color: #2ecc71;
    font-size: 22px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
    <h2>🩺 Diabetes Risk Predictor</h2>
    <p>Smart AI-based Health Analysis</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📋 Input Details")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 110)
bp = st.sidebar.slider("Blood Pressure", 0, 150, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.slider("DPF", 0.0, 3.0, 0.5)
age = st.sidebar.slider("Age", 1, 100, 30)

# ---------------- LAYOUT ----------------
left, right = st.columns([2, 1])

# ---------------- LEFT PANEL ----------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Patient Summary")

    st.write(f"**Pregnancies:** {pregnancies}")
    st.write(f"**Glucose:** {glucose}")
    st.write(f"**Blood Pressure:** {bp}")
    st.write(f"**BMI:** {bmi}")
    st.write(f"**Age:** {age}")

    if st.button("🔍 Predict"):
        data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        st.subheader("🧠 Result")

        if pred == 1:
            st.markdown(f'<p class="result-high">⚠️ High Risk ({prob:.2%})</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="result-low">✅ Low Risk ({prob:.2%})</p>', unsafe_allow_html=True)

        st.session_state["prob"] = prob

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT PANEL ----------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📈 Risk Level")

    prob = st.session_state.get("prob", 0)
    st.progress(int(prob * 100))
    st.write(f"### {prob:.2%}")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("ℹ️ Info")

    st.write("""
    • Glucose & BMI are major indicators  
    • Age plays important role  
    • Model is trained on medical dataset  

    ⚠️ Not a medical diagnosis
    """)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Made with ❤️ | Streamlit App")