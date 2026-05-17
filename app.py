import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Finder — Clinical Severity Assessment",
    page_icon="🦟",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 3-Color Design System
#   1. Deep Navy   (#0B1A2E)
#   2. Emerald     (#00C896)
#   3. Warm Amber  (#FFB347)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(160deg, #0B1A2E 0%, #112240 50%, #0B1A2E 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2137 0%, #091729 100%) !important;
        border-right: 1px solid rgba(0,200,150,0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #E0E0E0 !important;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 2.5rem 1rem 1rem;
    }
    .app-logo { font-size: 3.2rem; margin-bottom: 0.3rem; }
    .app-title {
        font-size: 2.6rem; font-weight: 900; letter-spacing: -1px;
        background: linear-gradient(135deg, #00C896 0%, #00E6A8 40%, #FFB347 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; line-height: 1.2;
    }
    .app-subtitle {
        font-size: 1rem; color: rgba(224,224,224,0.7); margin-top: 0.3rem;
        font-weight: 300; letter-spacing: 2px; text-transform: uppercase;
    }
    .divider-line {
        width: 80px; height: 3px;
        background: linear-gradient(90deg, #00C896, #FFB347);
        border-radius: 3px; margin: 1.2rem auto 2rem;
    }

    /* ── Glass Cards ── */
    .glass {
        background: rgba(255,255,255,0.04);
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0,200,150,0.12); border-radius: 18px;
        padding: 1.8rem 2rem; margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .glass:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,200,150,0.12);
    }
    .glass-label {
        color: #FFB347; font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2.5px; margin-bottom: 0.8rem;
    }
    .glass-heading {
        color: #FFFFFF; font-size: 1.15rem; font-weight: 600; margin-bottom: 0.5rem;
    }

    /* ── Severity Badges ── */
    .severity-badge {
        display: inline-block; padding: 0.6rem 2rem; border-radius: 50px;
        font-weight: 800; font-size: 1.1rem; letter-spacing: 1px;
        text-transform: uppercase; margin: 0.5rem 0;
    }
    .badge-mild {
        background: linear-gradient(135deg, #00C896, #00E6A8); color: #0B1A2E;
        box-shadow: 0 4px 20px rgba(0,200,150,0.35);
    }
    .badge-moderate {
        background: linear-gradient(135deg, #FFB347, #FFCA7A); color: #0B1A2E;
        box-shadow: 0 4px 20px rgba(255,179,71,0.35);
    }
    .badge-severe {
        background: linear-gradient(135deg, #FF4D6A, #FF6B81); color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(255,77,106,0.35);
    }
    .badge-uninfected {
        background: linear-gradient(135deg, #00C896, #00E6A8); color: #0B1A2E;
        box-shadow: 0 4px 20px rgba(0,200,150,0.35);
    }
    .badge-parasitized {
        background: linear-gradient(135deg, #FF4D6A, #FF6B81); color: #FFFFFF;
        box-shadow: 0 4px 20px rgba(255,77,106,0.35);
    }

    /* ── Result Container ── */
    .result-box {
        text-align: center; padding: 2.5rem 2rem;
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0,200,150,0.15); border-radius: 22px;
        margin: 1.5rem 0; box-shadow: 0 12px 48px rgba(0,0,0,0.3);
        animation: fadeUp 0.6s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .result-title {
        color: rgba(255,255,255,0.6); font-size: 0.75rem;
        text-transform: uppercase; letter-spacing: 3px;
        font-weight: 600; margin-bottom: 1rem;
    }
    .result-confidence {
        color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 1rem;
    }
    .result-confidence span { color: #00C896; font-weight: 700; }

    /* ── Recommendation Card ── */
    .rec-card {
        background: rgba(0,200,150,0.06);
        border: 1px solid rgba(0,200,150,0.18);
        border-radius: 14px; padding: 1.3rem 1.5rem;
        margin-top: 1rem; animation: fadeUp 0.8s ease-out;
    }
    .rec-card.warn {
        background: rgba(255,179,71,0.06); border-color: rgba(255,179,71,0.18);
    }
    .rec-card.danger {
        background: rgba(255,77,106,0.06); border-color: rgba(255,77,106,0.18);
    }
    .rec-title { font-weight: 700; font-size: 0.9rem; margin-bottom: 0.4rem; }
    .rec-title.green { color: #00C896; }
    .rec-title.amber { color: #FFB347; }
    .rec-title.red   { color: #FF4D6A; }
    .rec-text {
        color: rgba(224,224,224,0.8); font-size: 0.85rem; line-height: 1.6;
    }

    /* ── Stat Pills ── */
    .stat-pill {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(0,200,150,0.12);
        border-radius: 12px; padding: 0.9rem 1.3rem;
        text-align: center; min-width: 100px;
    }
    .stat-val { font-size: 1.5rem; font-weight: 800; color: #00C896; }
    .stat-lbl {
        font-size: 0.65rem; color: rgba(224,224,224,0.5);
        text-transform: uppercase; letter-spacing: 1.5px; margin-top: 0.2rem;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background: rgba(255,255,255,0.03);
        border-radius: 14px; padding: 6px;
        border: 1px solid rgba(0,200,150,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; color: rgba(224,224,224,0.6);
        font-weight: 600; padding: 10px 24px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,200,150,0.15) !important;
        color: #00C896 !important;
        border-bottom: none !important;
    }

    /* ── File Uploader ── */
    .stFileUploader > div > div {
        background: rgba(0,0,0,0.2);
        border: 2px dashed rgba(255,255,255,0.25);
        border-radius: 15px;
    }
    .stFileUploader > div > div:hover { border-color: #00C896; }

    /* ── Streamlit Overrides ── */
    .stSelectbox label, .stNumberInput label, .stRadio label {
        color: #E0E0E0 !important; font-weight: 500 !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00C896 0%, #00B386 100%) !important;
        color: #0B1A2E !important; font-weight: 700 !important;
        font-size: 1rem !important; border: none !important;
        border-radius: 12px !important; padding: 0.75rem 2rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,200,150,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0,200,150,0.45) !important;
    }
    div[data-testid="stProgress"] > div > div > div {
        background: linear-gradient(90deg, #00C896, #FFB347) !important;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="app-logo">🦟</div>
    <h1 class="app-title">Malaria Finder</h1>
    <p class="app-subtitle">Clinical Severity Assessment System</p>
    <div class="divider-line"></div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Model Loaders
# ─────────────────────────────────────────────
DATA_PATH = os.path.join('data', 'malaria_dataset.csv')
IMAGE_MODEL_PATH = 'malaria_resnet18.pth'
CLASS_NAMES_IMG = ['Parasitized', 'Uninfected']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def train_clinical_model():
    """Loads the clinical CSV dataset and trains a Random Forest classifier."""
    if not os.path.exists(DATA_PATH):
        return None, None, None, None

    df = pd.read_csv(DATA_PATH)
    le_gender = LabelEncoder()
    le_location = LabelEncoder()
    le_severity = LabelEncoder()

    df['Gender'] = le_gender.fit_transform(df['Gender'])
    df['Location'] = le_location.fit_transform(df['Location'])
    df['Severity'] = le_severity.fit_transform(df['Severity'])

    X = df.drop('Severity', axis=1)
    y = df['Severity']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    encoders = {'gender': le_gender, 'location': le_location, 'severity': le_severity}
    return model, encoders, acc, df


@st.cache_resource
def load_image_model():
    """Loads the trained ResNet-18 PyTorch model for cell-image classification."""
    if not os.path.exists(IMAGE_MODEL_PATH):
        return None

    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, len(CLASS_NAMES_IMG))
    )
    model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def process_cell_image(image):
    """Applies transforms matching train.py validation pipeline."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)


clinical_model, encoders, clinical_acc, df = train_clinical_model()
image_model = load_image_model()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Dataset Overview")
    if df is not None:
        raw_df = pd.read_csv(DATA_PATH)
        st.markdown(f"**Clinical Records:** {len(raw_df):,}")
        st.markdown("**Severity Distribution:**")
        counts = raw_df['Severity'].value_counts()
        for sev in ['Mild', 'Moderate', 'Severe']:
            if sev in counts.index:
                st.markdown(f"- {sev}: **{counts[sev]:,}** cases")
        if clinical_acc:
            st.markdown(f"**Symptom-Model Accuracy:** `{clinical_acc*100:.1f}%`")
    else:
        st.warning("Clinical dataset not found.")

    st.markdown("---")
    st.markdown("### 🔬 Image Model")
    if image_model is not None:
        st.success("ResNet-18 model loaded ✓")
    else:
        st.info(
            f"`{IMAGE_MODEL_PATH}` not found. "
            "Run `python train.py` to train the cell-image classifier."
        )

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Malaria Finder** combines clinical symptom analysis "
        "with deep-learning cell-image classification to provide "
        "a comprehensive malaria assessment toolkit for healthcare workers."
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:rgba(224,224,224,0.35);"
        "font-size:0.7rem;'>© 2026 Malaria Finder</p>",
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# Tabs — two analysis modes
# ─────────────────────────────────────────────
tab_symptoms, tab_image = st.tabs([
    "🩺  Clinical Symptom Assessment",
    "🔬  Cell Image Analysis"
])

# ══════════════════════════════════════════════
# TAB 1 — Clinical Symptom Assessment
# ══════════════════════════════════════════════
with tab_symptoms:
    st.markdown("""
    <div class="glass">
        <p class="glass-label">Patient Information</p>
        <p class="glass-heading">Enter the patient's clinical symptoms and demographics below</p>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("##### 🩺 Symptoms Observed")
        fever = st.selectbox("Fever", ["No", "Yes"], help="Is the patient experiencing fever?")
        headache = st.selectbox("Headache", ["No", "Yes"], help="Does the patient report headaches?")
        chills = st.selectbox("Chills", ["No", "Yes"], help="Is the patient experiencing chills or shivering?")
        fatigue = st.selectbox("Fatigue", ["No", "Yes"], help="Does the patient show signs of fatigue or weakness?")

    with col_b:
        st.markdown("##### 👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        weight = st.number_input("Weight (kg)", min_value=5, max_value=200, value=60, step=1)
        location = st.selectbox("Location", ["Urban", "Rural"], help="Where does the patient reside?")

    st.markdown("")

    if st.button("🔍  Assess Malaria Severity", use_container_width=True, key="btn_symptoms"):
        if clinical_model is None:
            st.error("⚠️ Clinical dataset not found. Ensure `data/malaria_dataset.csv` exists.")
        else:
            with st.spinner("Analysing clinical indicators…"):
                time.sleep(1.2)
                features = np.array([[
                    1 if fever == "Yes" else 0,
                    1 if headache == "Yes" else 0,
                    1 if chills == "Yes" else 0,
                    1 if fatigue == "Yes" else 0,
                    encoders['gender'].transform([gender])[0],
                    age, weight,
                    encoders['location'].transform([location])[0]
                ]])
                prediction_idx = clinical_model.predict(features)[0]
                probabilities = clinical_model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
                severity = encoders['severity'].inverse_transform([prediction_idx])[0]

            badge_map = {"Mild": ("badge-mild", "✅"), "Moderate": ("badge-moderate", "⚠️"), "Severe": ("badge-severe", "🚨")}
            badge_cls, icon = badge_map.get(severity, ("badge-moderate", "⚠️"))

            st.markdown(f"""
            <div class="result-box">
                <p class="result-title">Assessment Result</p>
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
                <span class="severity-badge {badge_cls}">{severity}</span>
                <p class="result-confidence">Model confidence: <span>{confidence*100:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown('<div class="glass"><p class="glass-label">Probability Breakdown</p></div>', unsafe_allow_html=True)
            class_names = encoders['severity'].classes_
            prob_cols = st.columns(len(class_names))
            for i, cls in enumerate(class_names):
                with prob_cols[i]:
                    pct = probabilities[i] * 100
                    st.markdown(f'<div class="stat-pill" style="width:100%"><div class="stat-val">{pct:.1f}%</div><div class="stat-lbl">{cls}</div></div>', unsafe_allow_html=True)
                    st.progress(probabilities[i])

            # Clinical recommendation
            recs = {
                "Mild": (
                    "", "green", "📋 Clinical Recommendation — Mild Case",
                    "The patient presents with mild indicators. Prescribe standard antimalarial therapy (e.g., ACT) and schedule a follow-up within 72 hours to monitor recovery. Advise the patient to stay hydrated, rest, and use insecticide-treated bed nets."
                ),
                "Moderate": (
                    " warn", "amber", "⚠️ Clinical Recommendation — Moderate Case",
                    "Multiple symptom indicators are present. Initiate supervised antimalarial treatment immediately and consider blood-film microscopy or RDT for parasitaemia confirmation. Monitor for progression to severe malaria within 24–48 hours."
                ),
                "Severe": (
                    " danger", "red", "🚨 Clinical Recommendation — Severe Case",
                    "This patient exhibits indicators consistent with severe malaria. Immediate hospitalisation is strongly recommended. Administer parenteral artesunate per WHO guidelines and monitor vital signs continuously. Refer to the nearest facility equipped for intensive care if complications arise."
                ),
            }
            card_cls, title_cls, title, text = recs[severity]
            st.markdown(f"""
            <div class="rec-card{card_cls}">
                <p class="rec-title {title_cls}">{title}</p>
                <p class="rec-text">{text}</p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — Cell Image Analysis
# ══════════════════════════════════════════════
with tab_image:
    st.markdown("""
    <div class="glass">
        <p class="glass-label">Microscopy Analysis</p>
        <p class="glass-heading">Upload a thin blood-smear cell image for infection detection</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a Cell Image (PNG / JPG)",
        type=['png', 'jpg', 'jpeg'],
        key="cell_upload"
    )

    if uploaded_file is not None:
        cell_image = Image.open(uploaded_file).convert('RGB')
        st.image(cell_image, caption="Uploaded Cell Image", use_container_width=True)

        if st.button("🔬  Analyse Cell Image", use_container_width=True, key="btn_image"):
            with st.spinner("Running deep-learning inference on cell image…"):
                time.sleep(1.2)

                if image_model is None:
                    st.warning(
                        f"**Model file `{IMAGE_MODEL_PATH}` not found.** "
                        "Please run `python train.py` to train the ResNet-18 model first. "
                        "Showing a demo prediction below."
                    )
                    prediction = "Parasitized" if "parasite" in uploaded_file.name.lower() else "Uninfected"
                    confidence = 0.97
                    prob_para, prob_uninf = (0.97, 0.03) if prediction == "Parasitized" else (0.03, 0.97)
                else:
                    input_tensor = process_cell_image(cell_image)
                    with torch.no_grad():
                        outputs = image_model(input_tensor)
                        probs = torch.nn.functional.softmax(outputs[0], dim=0)
                        confidence, predicted_idx = torch.max(probs, 0)
                        prediction = CLASS_NAMES_IMG[predicted_idx.item()]
                        confidence = confidence.item()
                        prob_para = probs[0].item()
                        prob_uninf = probs[1].item()

            if prediction == "Uninfected":
                badge_cls, icon = "badge-uninfected", "✅"
            else:
                badge_cls, icon = "badge-parasitized", "🦠"

            st.markdown(f"""
            <div class="result-box">
                <p class="result-title">Cell Analysis Result</p>
                <div style="font-size:2.5rem;margin-bottom:0.5rem;">{icon}</div>
                <span class="severity-badge {badge_cls}">{prediction}</span>
                <p class="result-confidence">Model confidence: <span>{confidence*100:.1f}%</span></p>
            </div>
            """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown('<div class="glass"><p class="glass-label">Probability Breakdown</p></div>', unsafe_allow_html=True)
            pc, pu = st.columns(2)
            with pc:
                st.markdown(f'<div class="stat-pill" style="width:100%"><div class="stat-val">{prob_para*100:.1f}%</div><div class="stat-lbl">Parasitized</div></div>', unsafe_allow_html=True)
                st.progress(prob_para)
            with pu:
                st.markdown(f'<div class="stat-pill" style="width:100%"><div class="stat-val">{prob_uninf*100:.1f}%</div><div class="stat-lbl">Uninfected</div></div>', unsafe_allow_html=True)
                st.progress(prob_uninf)

            # Recommendation
            if prediction == "Uninfected":
                st.markdown("""
                <div class="rec-card">
                    <p class="rec-title green">✅ No Malaria Parasite Detected</p>
                    <p class="rec-text">
                        The cell image does not show signs of <em>Plasmodium</em> infection.
                        If clinical symptoms persist, consider repeating the test with a fresh
                        blood smear or performing a Rapid Diagnostic Test (RDT) for confirmation.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="rec-card danger">
                    <p class="rec-title red">🦠 Malaria Parasite Detected</p>
                    <p class="rec-text">
                        The cell image indicates the presence of a <em>Plasmodium</em> parasite.
                        Begin antimalarial treatment immediately per local clinical guidelines.
                        Confirm species identification through microscopy and assess severity
                        using the Clinical Symptom Assessment tab for a complete diagnosis.
                    </p>
                </div>
                """, unsafe_allow_html=True)
