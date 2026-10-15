import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st  # pyright: ignore[reportMissingImports]
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

st.set_page_config(
    page_title="US Accident Severity Predictor",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


class AccidentSeverityNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_preprocessing():
    return joblib.load("accident_preprocessing.pkl")


@st.cache_resource
def load_bert():
    tokenizer = AutoTokenizer.from_pretrained("accident_distilbert_model")
    bert_model = AutoModel.from_pretrained("accident_distilbert_model")
    bert_model.eval()
    return tokenizer, bert_model


preprocessing = load_preprocessing()
tokenizer, bert_model = load_bert()
label_encoder = preprocessing["label_encoder"]

model = AccidentSeverityNN(
    input_size=803,
    num_classes=len(label_encoder.classes_),
)
model.load_state_dict(
    torch.load(
        "accident_severity_model.pth",
        map_location="cpu",
        weights_only=True,
    )
)
model.eval()


def extract_text_features(text):
    inputs = tokenizer(
        [text],
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=64,
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()


def prepare_structured_features(input_data):
    input_df = pd.DataFrame([input_data])
    numerical_features = preprocessing["numeric_features"]
    categorical_features = preprocessing["categorical_features"]
    numerical_data = input_df[numerical_features]
    categorical_data = input_df[categorical_features]

    numerical_imputed = preprocessing["num_imputer"].transform(numerical_data)
    categorical_imputed = preprocessing["cat_imputer"].transform(categorical_data)
    categorical_encoded = preprocessing["encoder"].transform(categorical_imputed)
    numerical_scaled = preprocessing["scaler"].transform(numerical_imputed)

    return np.concatenate([numerical_scaled, categorical_encoded], axis=1)


def predict_severity(description, input_data):
    text_features = extract_text_features(description)
    structured_features = prepare_structured_features(input_data)

    combined_features = np.concatenate(
        [text_features, structured_features],
        axis=1,
    )
    input_tensor = torch.tensor(
        combined_features,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_index = torch.argmax(probabilities, dim=1).item()

    print("Class order:", label_encoder.classes_)
    print("Class scores:", outputs.numpy())
    print("Class probabilities:", probabilities.numpy())

    return label_encoder.inverse_transform([predicted_index])[0]
    return label_encoder.inverse_transform([predicted_index])[0]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --page: #06111d;
        --panel: #0c1b2b;
        --panel-raised: #10243a;
        --line: #203b56;
        --muted: #91a8bf;
        --text: #edf5fc;
        --blue: #3297f6;
        --green: #42dc98;
        --amber: #ffc247;
        --orange: #ff8745;
        --red: #ff525d;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', 'Segoe UI', sans-serif;
        letter-spacing: 0;
    }
    .stApp {
        color: var(--text);
        background:
            radial-gradient(ellipse at 50% -25%, #16304a 0%, transparent 52%),
            var(--page);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1480px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1, h2, h3, h4, p, label { color: var(--text); }
    h1, h2, h3, h4 { font-family: 'Manrope', 'Segoe UI', sans-serif; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(12, 27, 43, 0.96);
        border: 1px solid var(--line);
        border-radius: 8px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1rem;
    }
    [data-testid="stWidgetLabel"] p {
        color: #c6d6e5;
        font-size: 0.78rem;
        font-weight: 600;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #0a1929;
        border-color: #2b4865;
        color: var(--text);
        border-radius: 6px;
    }
    [data-testid="stTextArea"] textarea::placeholder { color: #7e94aa; }
    [data-testid="stNumberInput"] button { color: #9bc8f5; }
    .stButton > button {
        min-height: 44px;
        color: white;
        background: #278cf0;
        border: 1px solid #4ba6ff;
        border-radius: 6px;
        font-weight: 700;
        transition: background 150ms ease, transform 150ms ease;
    }
    .stButton > button:hover {
        background: #1476dc;
        border-color: #70b9ff;
        color: white;
        transform: translateY(-1px);
    }
    .hero {
        position: relative;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        min-height: 142px;
        margin-bottom: 1rem;
        padding: 1.3rem 1.6rem;
        overflow: hidden;
        border: 1px solid #294663;
        border-radius: 8px;
        background-image:
            linear-gradient(90deg, rgba(5, 17, 30, .96) 0%, rgba(5, 17, 30, .78) 48%, rgba(5, 17, 30, .15) 100%),
            url('https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?auto=format&fit=crop&w=1800&q=85');
        background-size: cover;
        background-position: center 58%;
    }
    .hero-copy { position: relative; z-index: 1; }
    .hero-kicker {
        margin: 0 0 .22rem;
        color: #9cc9f6;
        font-size: .72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .08em;
    }
    .hero h1 {
        margin: 0;
        color: #f6fbff;
        font-size: 1.72rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .hero-subtitle {
        margin: .34rem 0 0;
        color: #d1deeb;
        font-size: .88rem;
    }
    .hero-meta {
        margin: .4rem 0 0;
        color: #a7c9e7;
        font-size: .72rem;
    }
    .hero-mark {
        position: relative;
        z-index: 1;
        display: grid;
        width: 48px;
        height: 48px;
        flex: 0 0 48px;
        place-items: center;
        border: 1px solid rgba(167, 201, 231, .32);
        border-radius: 8px;
        background: rgba(5, 17, 30, .58);
        font-size: 1.65rem;
    }
    .panel-title {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: .5rem;
        margin: 0 0 .7rem;
        color: #eaf4ff;
        font: 700 1rem 'Manrope', 'Segoe UI', sans-serif;
    }
    .panel-tag {
        color: #7894ad;
        font: 700 .63rem 'DM Sans', sans-serif;
        letter-spacing: .07em;
    }
    .section-title {
        display: flex;
        align-items: center;
        gap: .5rem;
        margin: .52rem 0 .38rem;
        padding-top: .5rem;
        border-top: 1px solid rgba(45, 70, 96, .72);
        color: #a8d3fb;
        font: 700 .82rem 'Manrope', 'Segoe UI', sans-serif;
    }
    .section-icon { font-size: 1rem; }
    .result-heading {
        display: flex;
        align-items: center;
        gap: .55rem;
        margin-bottom: .75rem;
        color: var(--green);
        font: 700 1rem 'Manrope', 'Segoe UI', sans-serif;
    }
    .checkmark {
        display: grid;
        width: 24px;
        height: 24px;
        place-items: center;
        border-radius: 50%;
        background: var(--green);
        color: #072219;
        font-weight: 800;
    }
    .result-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        min-height: 108px;
        margin-bottom: .72rem;
        padding: .85rem 1rem;
        border: 1px solid color-mix(in srgb, var(--severity-color) 48%, #203b56);
        border-radius: 7px;
        background:
            linear-gradient(105deg, rgba(8, 27, 39, .98) 0%, rgba(8, 27, 39, .88) 64%, rgba(8, 27, 39, .46) 100%),
            url('https://images.unsplash.com/photo-1449965408869-eaa3f722e40d?auto=format&fit=crop&w=900&q=75') center 60% / cover;
    }
    .result-card.empty { --severity-color: #3c759f; }
    .result-level {
        display: inline-block;
        margin: 0 0 .35rem;
        padding: .25rem .72rem;
        border-radius: 6px;
        background: var(--severity-color);
        color: #071622;
        font: 800 1.16rem 'Manrope', 'Segoe UI', sans-serif;
    }
    .result-level.empty-level { background: #17334b; color: #c2d8eb; }
    .result-name { margin: 0; color: var(--severity-color); font-size: .74rem; font-weight: 700; }
    .result-copy { margin: .25rem 0 0; color: #d6e2ee; font-size: .72rem; }
    .result-art {
        display: grid;
        width: 76px;
        height: 76px;
        flex: 0 0 76px;
        place-items: center;
        border: 1px solid rgba(198, 220, 240, .24);
        border-radius: 50%;
        background: rgba(8, 20, 33, .72);
        font-size: 2rem;
    }
    .legend-title, .overview-title {
        margin: .72rem 0 .42rem;
        color: #bbd3e9;
        font: 700 .77rem 'Manrope', 'Segoe UI', sans-serif;
    }
    .severity-legend {
        overflow: hidden;
        border: 1px solid #1d3852;
        border-radius: 6px;
        background: #091827;
    }
    .severity-row {
        display: grid;
        grid-template-columns: 22px 82px 1fr;
        align-items: center;
        gap: .45rem;
        min-height: 29px;
        padding: .25rem .55rem;
        border-bottom: 1px solid rgba(34, 60, 84, .7);
        color: #a9bfd3;
        font-size: .68rem;
    }
    .severity-row:last-child { border-bottom: 0; }
    .severity-row.active { background: rgba(50, 151, 246, .12); color: #edf5fc; }
    .severity-dot {
        display: grid;
        width: 15px;
        height: 15px;
        place-items: center;
        border-radius: 50%;
        color: #071622;
        font-size: .59rem;
        font-weight: 800;
    }
    .severity-row strong { color: #e4eff8; font-size: .68rem; }
    .model-flow {
        display: grid;
        grid-template-columns: 1fr 15px 1fr 15px 1fr 15px .85fr;
        align-items: center;
        gap: .26rem;
        padding: .6rem;
        border: 1px solid #1d3852;
        border-radius: 6px;
        background: #091827;
    }
    .flow-node {
        display: grid;
        min-height: 52px;
        align-content: center;
        justify-items: center;
        padding: .35rem .2rem;
        border: 1px solid #254764;
        border-radius: 5px;
        background: #10243a;
        color: #e1eef9;
        text-align: center;
        font-size: .61rem;
        font-weight: 700;
    }
    .flow-node span { margin-top: .12rem; color: #8ea9c2; font-size: .54rem; font-weight: 500; }
    .flow-node.output { border-color: #24634e; background: #0e2a29; color: #74e8b4; }
    .flow-arrow { color: #70aee8; text-align: center; font-size: .9rem; }
    .empty-state {
        display: flex;
        align-items: center;
        gap: .75rem;
        min-height: 108px;
        margin-bottom: .72rem;
        padding: .85rem 1rem;
        border: 1px solid #1f3c58;
        border-radius: 7px;
        background: linear-gradient(110deg, #0b1d2d, #102a40);
    }
    .empty-state-icon {
        display: grid;
        width: 42px;
        height: 42px;
        flex: 0 0 42px;
        place-items: center;
        border: 1px solid #315979;
        border-radius: 50%;
        color: #65b4ff;
        font-size: 1.2rem;
    }
    .empty-state h3 { margin: 0 0 .2rem; font-size: .92rem; }
    .empty-state p { margin: 0; color: #a8bfd3; font-size: .72rem; }
    .lower-card {
        min-height: 108px;
        padding: .85rem 1rem;
        border: 1px solid var(--line);
        border-radius: 7px;
        background: #0b1a2a;
    }
    .lower-card h3 { margin: 0 0 .35rem; font-size: .8rem; }
    .lower-card p { margin: 0; color: #a9bfd3; font-size: .68rem; line-height: 1.5; }
    .dataset-badges { display: flex; flex-wrap: wrap; gap: .35rem; margin-top: .55rem; }
    .dataset-badge {
        padding: .22rem .45rem;
        border: 1px solid #254764;
        border-radius: 4px;
        background: #10243a;
        color: #a8c8e5;
        font-size: .61rem;
    }
    .footer-line {
        display: flex;
        justify-content: space-between;
        gap: .6rem;
        margin-top: .65rem;
        padding: .55rem .1rem 0;
        border-top: 1px solid #1b3045;
        color: #8099b0;
        font-size: .63rem;
    }
    .footer-heart { color: #ff5865; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: .55rem .55rem 1rem; }
        .hero { min-height: 126px; padding: 1rem; background-position: 63% center; }
        .hero h1 { max-width: 28rem; font-size: 1.25rem; }
        .hero-mark { width: 40px; height: 40px; flex-basis: 40px; font-size: 1.3rem; }
        .model-flow { grid-template-columns: 1fr 10px 1fr 10px 1fr 10px .85fr; gap: .12rem; }
        .flow-node { min-height: 48px; font-size: .54rem; }
        .flow-node span { font-size: .48rem; }
        .severity-row { grid-template-columns: 20px 76px 1fr; font-size: .63rem; }
        .footer-line { flex-wrap: wrap; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="hero">
        <div class="hero-copy">
            <p class="hero-kicker">United States · Road safety intelligence</p>
            <h1>AI-Powered Accident Severity Prediction</h1>
            <p class="hero-subtitle">Assess a road incident with deep learning and DistilBERT.</p>
            <p class="hero-meta">Accident description · Weather · Road conditions</p>
        </div>
        <div class="hero-mark" aria-hidden="true">🚘</div>
    </header>
    """,
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([0.94, 1.06], gap="medium")
prediction_error = None

with left_col:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-title"><span>Accident Details</span><span class="panel-tag">INPUT</span></div>',
            unsafe_allow_html=True,
        )
        description = st.text_area(
            "Accident Description",
            placeholder="e.g., A car collided with a truck at an intersection. The weather was clear and the road was dry.",
            height=92,
            max_chars=500,
            label_visibility="visible",
        )

        st.markdown(
            '<div class="section-title"><span class="section-icon">☁</span>Weather Information</div>',
            unsafe_allow_html=True,
        )
        weather_index = preprocessing["categorical_features"].index("Weather_Condition")
        weather_options = list(preprocessing["encoder"].categories_[weather_index])
        weather = st.selectbox("Weather Condition", weather_options)

        weather_col1, weather_col2 = st.columns(2, gap="small")
        with weather_col1:
            temperature = st.number_input(
                "Temperature (°F)", min_value=-50.0, max_value=150.0, value=75.0
            )
            humidity = st.number_input(
                "Humidity (%)", min_value=0.0, max_value=100.0, value=60.0
            )
        with weather_col2:
            visibility = st.number_input(
                "Visibility (miles)", min_value=0.0, max_value=100.0, value=10.0
            )
            wind_speed = st.number_input(
                "Wind Speed (mph)", min_value=0.0, max_value=200.0, value=5.0
            )

        st.markdown(
            '<div class="section-title"><span class="section-icon">🛣</span>Road &amp; Location Details</div>',
            unsafe_allow_html=True,
        )
        road_col1, road_col2 = st.columns(2, gap="small")
        with road_col1:
            sunrise_sunset = st.selectbox("Time of Day", ["Day", "Night"])
            traffic_signal = st.selectbox("Traffic Signal", ["Yes", "No"])
        with road_col2:
            crossing = st.selectbox("Crossing Present", ["Yes", "No"])
            junction = st.selectbox("Junction Present", ["Yes", "No"])

        predict_button = st.button(
            "▥   Predict Severity",
            use_container_width=True,
        )

if predict_button:
    if not description.strip():
        st.session_state.pop("prediction", None)
        prediction_error = "Add a short accident description to run the prediction."
    else:
        input_data = {
            "Temperature(F)": temperature,
            "Humidity(%)": humidity,
            "Visibility(mi)": visibility,
            "Weather_Condition": weather,
            "Wind_Speed(mph)": wind_speed,
            "Sunrise_Sunset": sunrise_sunset,
            "Traffic_Signal": traffic_signal == "Yes",
            "Crossing": crossing == "Yes",
            "Junction": junction == "Yes",
        }
        try:
            st.session_state["prediction"] = predict_severity(description, input_data)
        except Exception as error:
            st.session_state.pop("prediction", None)
            prediction_error = str(error)

with right_col:
    with st.container(border=True):
        st.markdown(
            '<div class="result-heading"><span class="checkmark">✓</span>Prediction Result</div>',
            unsafe_allow_html=True,
        )

        prediction = st.session_state.get("prediction")
        if prediction_error:
            st.error(f"Prediction error: {prediction_error}")
        elif prediction is None:
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">ⓘ</div>
                    <div>
                        <h3>Ready when you are</h3>
                        <p>Enter the incident details and predict a severity level.</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            try:
                severity_number = int(prediction)
            except (TypeError, ValueError):
                severity_number = None

            severity_info = {
                1: ("Minor", "No injury or possible injury.", "#42dc98"),
                2: ("Moderate", "Non-incapacitating injury.", "#ffc247"),
                3: ("Serious", "Incapacitating injury.", "#ff8745"),
                4: ("Fatal", "Fatality.", "#ff525d"),
            }
            severity_name, severity_description, severity_color = severity_info.get(
                severity_number,
                ("Recorded category", "Model output based on the supplied details.", "#68b6ff"),
            )
            severity_text = html.escape(str(prediction))
            st.markdown(
                f"""
                <div class="result-card" style="--severity-color:{severity_color}">
                    <div>
                        <div class="result-level">Severity {severity_text}</div>
                        <p class="result-name">{severity_name}</p>
                        <p class="result-copy">{severity_description}</p>
                    </div>
                    <div class="result-art" aria-hidden="true">🚨</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Model estimate only. It does not guarantee the actual impact of an incident."
            )

        severity_rows = []
        severity_legend = [
            (1, "Minor", "No injury or possible injury.", "#42dc98"),
            (2, "Moderate", "Non-incapacitating injury.", "#ffc247"),
            (3, "Serious", "Incapacitating injury.", "#ff8745"),
            (4, "Fatal", "Fatality.", "#ff525d"),
        ]
        try:
            active_severity = int(prediction) if prediction is not None else None
        except (TypeError, ValueError):
            active_severity = None
        for level, name, description_text, color in severity_legend:
            active_class = " active" if level == active_severity else ""
            severity_rows.append(
                f'<div class="severity-row{active_class}">'
                f'<span class="severity-dot" style="background:{color}">{level}</span>'
                f"<strong>{name}</strong><span>{description_text}</span></div>"
            )
        st.markdown(
            '<div class="legend-title">Severity Levels <span style="color:#7894ad;font-weight:400">(US dataset)</span></div>'
            f'<div class="severity-legend">{"".join(severity_rows)}</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="overview-title">Model Overview</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="model-flow">
                <div class="flow-node">Accident Text<span>DistilBERT embedding</span></div>
                <div class="flow-arrow">›</div>
                <div class="flow-node">Weather &amp; Road<span>Structured features</span></div>
                <div class="flow-arrow">›</div>
                <div class="flow-node">Neural Network<span>Deep learning</span></div>
                <div class="flow-arrow">›</div>
                <div class="flow-node output">Severity<span>Classes 1–4</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

dataset_col, about_col = st.columns([1.55, 1], gap="medium")
with dataset_col:
    st.markdown(
        """
        <div class="lower-card">
            <h3>🇺🇸 US Road Accident Dataset</h3>
            <p>Trained on US accident records with incident descriptions, weather observations, and road-condition details.</p>
            <div class="dataset-badges">
                <span class="dataset-badge">▤ Real-world data</span>
                <span class="dataset-badge">☁ Weather conditions</span>
                <span class="dataset-badge">⌁ Road features</span>
                <span class="dataset-badge">⌖ Location context</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with about_col:
    st.markdown(
        """
        <div class="lower-card">
            <h3>ⓘ About This Model</h3>
            <p>DistilBERT analyzes accident text while a neural network combines it with structured weather and road features to estimate severity.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer-line">
        <span><span class="footer-heart">♥</span> &nbsp;Safer Roads &nbsp; | &nbsp; Smarter Predictions &nbsp; | &nbsp; Better Tomorrow</span>
        <span>Built with Streamlit</span>
    </div>
    """,
    unsafe_allow_html=True,
)