
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "meilleur_modele.joblib"

LABEL_MAP = {0: "Low", 1: "Medium", 2: "High"}

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Physical_Activity",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
    "Part_Time_Job",
    "Diet_Quality",
    "Previous_Mental_Health_History",
]

st.set_page_config(page_title="Student Mental Health Predictor")

st.title("Student Mental Health Predictor")
st.write("Enter student information to test the Machine Learning model.")
st.caption("Educational prototype only. Not a medical diagnosis or screening tool.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 0, 100, 20)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-Binary"])
        year_of_study = st.number_input("Year of study", 1, 8, 2)
        sleep_hours = st.number_input("Sleep hours per night", 0.0, 24.0, 7.0)
        study_hours = st.number_input("Study hours per week", 0.0, 100.0, 15.0)
        gpa = st.number_input("GPA", 0.0, 4.0, 3.0)
        social_media = st.number_input("Social media hours per day", 0.0, 24.0, 3.0)
        screen_time = st.number_input("Screen time hours per day", 0.0, 24.0, 6.0)

    with col2:
        physical_activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
        financial_stress = st.slider("Financial stress", 0.0, 10.0, 5.0)
        family_support = st.slider("Family support", 0.0, 10.0, 5.0)
        attendance = st.slider("Attendance rate (%)", 0, 100, 85)
        part_time_job = st.selectbox("Part-time job", ["No", "Yes"])
        diet_quality = st.selectbox("Diet quality", ["Poor", "Average", "Good"])
        previous_history = st.selectbox("Previous mental health history", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")


if submitted:

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Year_of_Study": year_of_study,
        "Sleep_Hours": sleep_hours,
        "Study_Hours_Per_Week": study_hours,
        "GPA": gpa,
        "Social_Media_Hours": social_media,
        "Screen_Time_Hours": screen_time,
        "Physical_Activity": physical_activity,
        "Financial_Stress": financial_stress,
        "Family_Support": family_support,
        "Attendance_Rate": attendance,
        "Part_Time_Job": part_time_job,
        "Diet_Quality": diet_quality,
        "Previous_Mental_Health_History": previous_history
    }])[FEATURE_COLUMNS]

    st.subheader("Student Information")
    st.dataframe(input_df)

    try:
        prediction = model.predict(input_df)[0]
        label = LABEL_MAP[int(prediction)]

        st.subheader("Prediction Result")
        st.metric("Predicted Class", label)

        probabilities = model.predict_proba(input_df)[0]

        st.subheader("Model Probabilities")

        for class_id, probability in zip(model.classes_, probabilities):
            class_name = LABEL_MAP[int(class_id)]
            st.write(f"{class_name}: {probability:.1%}")
            st.progress(float(probability))

        st.info(
            "These are model estimates for dataset categories, "
            "not clinical probabilities or a medical assessment."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
