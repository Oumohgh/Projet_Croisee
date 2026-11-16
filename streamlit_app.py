
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,ConfusionMatrixDisplay)

st.set_page_config(page_title="Student Mental Health Analysis",layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "meilleur_modele.joblib"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.csv"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.csv"

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
    "Previous_Mental_Health_History"]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title(" Student Mental Health Analysis")
st.caption(
    "Educational Machine Learning project. "
    "This application is not a medical diagnosis or screening tool."
)

page = st.sidebar.radio(
    "Navigation",
    ["Model Performance", "Make a Prediction"]
)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()


if page == "Model Performance":

    st.header("Model Performance")

    st.write(
        "This dashboard evaluates the trained model "
        "using the test dataset."
    )

    try:
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, average="macro", zero_division=0
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("Precision", f"{precision:.2%}")
        col3.metric("Recall", f"{recall:.2%}")
        col4.metric("F1-score", f"{f1:.2%}")

        st.divider()

        st.subheader("Performance Metrics Chart")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-score"],
            "Score": [accuracy, precision, recall, f1]
        })

        st.bar_chart(metrics_df.set_index("Metric"))

        st.divider()

        st.subheader("Confusion Matrix")

        fig, ax = plt.subplots(figsize=(6, 4))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            display_labels=["Low", "Medium", "High"],
            cmap="Blues",
            ax=ax
        )

        st.pyplot(fig)
        plt.close(fig)

        st.divider()

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2],
            target_names=["Low", "Medium", "High"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).T.round(3),
            use_container_width=True
        )

        st.info(
            "These metrics measure performance on the test dataset. "
            "They do not establish clinical validity."
        )

    except Exception as e:
        st.error(f"Unable to calculate model performance: {e}")


elif page == "Make a Prediction":

    st.header("Make a Prediction")

    st.write(
        "Enter student information to test "
        "the trained Machine Learning model."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 0, 100, 20)

            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Non-Binary"]
            )

            year_of_study = st.number_input(
                "Year of Study", 1, 8, 2
            )

            sleep_hours = st.number_input(
                "Sleep Hours per Night", 0.0, 24.0, 7.0
            )

            study_hours = st.number_input(
                "Study Hours per Week", 0.0, 100.0, 15.0
            )

            gpa = st.number_input(
                "GPA", 0.0, 4.0, 3.0
            )

            social_media = st.number_input(
                "Social Media Hours per Day", 0.0, 24.0, 3.0
            )

            screen_time = st.number_input(
                "Screen Time Hours per Day", 0.0, 24.0, 6.0
            )

        with col2:
            physical_activity = st.selectbox(
                "Physical Activity",
                ["Low", "Moderate", "High"]
            )

            financial_stress = st.slider(
                "Financial Stress", 0.0, 10.0, 5.0)

            family_support = st.slider(
                "Family Support", 0.0, 10.0, 5.0)

            attendance = st.slider(
                "Attendance Rate (%)", 0, 100, 85)

            part_time_job = st.selectbox(
                "Part-Time Job", ["No", "Yes"])

            diet_quality = st.selectbox(
                "Diet Quality", ["Poor", "Average", "Good"])

            previous_history = st.selectbox(
                "Previous Mental Health History",
                ["No", "Yes"])

        submitted = st.form_submit_button(
            "Predict",
            use_container_width=True)

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

        try:
            prediction = model.predict(input_df)[0]
            label = LABEL_MAP[int(prediction)]

            st.divider()
            st.subheader("Prediction Result")

            st.metric("Predicted Category", label)

            probabilities = model.predict_proba(input_df)[0]

            st.subheader("Model Probabilities")

            for class_id, probability in zip(
                model.classes_, probabilities
            ):
                class_name = LABEL_MAP[int(class_id)]
                st.write(f"{class_name}: {probability:.1%}")
                st.progress(float(probability))

            st.info(
                "The predicted category and probabilities are "
                "model outputs for educational demonstration only, "
                "not a clinical assessment."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
