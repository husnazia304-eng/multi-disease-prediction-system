import gradio as gr
import numpy as np
import joblib

# LOAD MODELS
models = {
    "diabetes": joblib.load("models/diabetes_model.pkl"),
    "heart": joblib.load("models/heart_model.pkl"),
    "parkinsons": joblib.load("models/parkinsons_model.pkl"),
    "liver": joblib.load("models/liver_model.pkl"),
}

# LOAD SCALERS
scalers = {
    "diabetes": joblib.load("models/diabetes_scaler.pkl"),
    "heart": joblib.load("models/heart_scaler.pkl"),
    "parkinsons": joblib.load("models/parkinsons_scaler.pkl"),
    "liver": joblib.load("models/liver_scaler.pkl"),
}

# FEATURE NAMES
feature_names = {
    "diabetes": [
        "Glucose",
        "BMI",
        "Age",
        "Blood Pressure",
        "Insulin"
    ],

    "heart": [
        "Age",
        "Cholesterol",
        "Blood Pressure",
        "Max Heart Rate",
        "ST Depression"
    ],

    "parkinsons": [
        "Tremor",
        "Voice Change",
        "Speech Rate",
        "Hand Movement",
        "Balance"
    ],

    "liver": [
        "Bilirubin",
        "Alkaline Phosphotase",
        "ALT",
        "AST",
        "Albumin"
    ]
}

# PREDICTION FUNCTION
def predict(disease, f1, f2, f3, f4, f5):

    model = models[disease]
    scaler = scalers[disease]

    x = np.array([[f1, f2, f3, f4, f5]])

    # SAME SCALING AS TRAINING
    x = scaler.transform(x)

    pred = model.predict(x)[0]

    prob = model.predict_proba(x)[0][1]

    # RISK LEVEL
    if prob > 0.7:
        risk = "High"

    elif prob > 0.4:
        risk = "Medium"

    else:
        risk = "Low"

    result = "Positive" if pred == 1 else "Negative"

    return result, f"{prob*100:.2f}%", risk


# UPDATE FEATURE LABELS
def update_labels(disease):

    names = feature_names[disease]

    return (
        gr.Number(label=names[0]),
        gr.Number(label=names[1]),
        gr.Number(label=names[2]),
        gr.Number(label=names[3]),
        gr.Number(label=names[4]),
    )



# GUI
with gr.Blocks() as app:

    gr.Markdown("# 🧠 Multi-Disease Intelligent Prediction System")

    disease = gr.Dropdown(
        choices=["diabetes", "heart", "parkinsons", "liver"],
        value="diabetes",
        label="Select Disease"
    )

    f1 = gr.Number(label="Glucose")
    f2 = gr.Number(label="BMI")
    f3 = gr.Number(label="Age")
    f4 = gr.Number(label="Blood Pressure")
    f5 = gr.Number(label="Insulin")

    # CHANGE LABELS AUTOMATICALLY
    disease.change(
        fn=update_labels,
        inputs=disease,
        outputs=[f1, f2, f3, f4, f5]
    )

    output1 = gr.Textbox(label="Prediction")
    output2 = gr.Textbox(label="Confidence")
    output3 = gr.Textbox(label="Risk Level")

    btn = gr.Button("Predict")

    btn.click(
        fn=predict,
        inputs=[disease, f1, f2, f3, f4, f5],
        outputs=[output1, output2, output3]
    )

app.launch(debug=True)
