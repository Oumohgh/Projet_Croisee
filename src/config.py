from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT / "data" / "orig" / "student_mental_health.csv"
MODELS_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
TARGET = "Mental_Health_Risk"
ID_COL = "Student_ID"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2}
RISK_LABELS = list(RISK_ORDER)       


QUANT_CONTINUES = ["Sleep_Hours", "Study_Hours_Per_Week", "GPA", "Social_Media_Hours",
                   "Screen_Time_Hours", "Financial_Stress", "Family_Support", "Attendance_Rate"]
QUANT_DISCRETES = ["Age", "Year_of_Study"]
ORDINAL_COLS = {"Physical_Activity": ["Low", "Moderate", "High"],
                "Diet_Quality": ["Poor", "Average", "Good"]}
NOMINAL_COLS = ["Gender", "Part_Time_Job", "Previous_Mental_Health_History"]