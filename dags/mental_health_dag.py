
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

from src.prepare_data import prepare_data
from src.preprocess_data import preprocess_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model


with DAG(
    dag_id="student_mental_health_dag",
    start_date=datetime(2026, 11, 13),
    schedule="@weekly",
    catchup=False
) as dag:

    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=prepare_data
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model
    )

    prepare >> preprocess >> train >> evaluate
