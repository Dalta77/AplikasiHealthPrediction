import streamlit as st
import joblib
import numpy as np

# Load models and encoders
@st.cache_resource
def load_models():
    try:
        cholesterol_model = joblib.load('model/cholesterol_model_tuned.pkl')
        drug_model = joblib.load('model/drug_model_tuned.pkl')
        scaler = joblib.load('model/scaler.pkl')
        label_encoder_cholesterol = joblib.load('model/label_encoder_cholesterol.pkl')
        label_encoder_drug = joblib.load('model/label_encoder_drug.pkl')
        return cholesterol_model, drug_model, scaler, label_encoder_cholesterol, label_encoder_drug
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        return None, None, None, None, None

cholesterol_model, drug_model, scaler, label_encoder_cholesterol, label_encoder_drug = load_models()

# BP mapping to numerical values
BP_MAPPING = {'low': 0, 'normal': 1, 'high': 2}


# Prediction function
def predict(model, scaler, encoder, features):
    try:
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)
        result = encoder.inverse_transform(prediction)
        return result[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# Streamlit app
def main():
    st.title("Health Prediction App")
    st.sidebar.title("Choose Prediction Type")

    prediction_type = st.sidebar.radio(
        "Select the type of prediction:",
        ("Cholesterol Level", "Drug Recommendation")
    )

    st.subheader(f"{prediction_type} Prediction")

    # Input fields
    age = st.number_input(
        "Age (in years):",
        min_value=0, max_value=120, value=25,
        help="Enter the age of the individual."
    )
    sex = st.radio(
        "Sex (0 for Female, 1 for Male):",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Select the sex of the individual."
    )
    bp = st.selectbox(
        "BP Level (Blood Pressure):",
        options=['low', 'normal', 'high'],
        help="Choose the blood pressure level: 'low', 'normal', or 'high'."
    )
    na_to_k = st.number_input(
        "Na_to_K (Sodium to Potassium Ratio):",
        min_value=0.0, max_value=100.0, value=10.0,
        help="Enter the sodium-to-potassium ratio (a floating-point number)."
    )

    # Predict button
    if st.button("Predict"):
        # Convert BP string to numerical value
        bp_numerical = BP_MAPPING[bp]
        features = [age, sex, bp_numerical, na_to_k]

        if prediction_type == "Cholesterol Level":
            if cholesterol_model:
                result = predict(cholesterol_model, scaler, label_encoder_cholesterol, features)
                st.success(f"Predicted Cholesterol Level: {result}")
            else:
                st.error("Cholesterol model not loaded.")
        elif prediction_type == "Drug Recommendation":
            if drug_model:
                result = predict(drug_model, scaler, label_encoder_drug, features)
                st.success(f"Recommended Drug: {result}")
            else:
                st.error("Drug model not loaded.")

if __name__ == "__main__":
    main()
