import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.http.operators.http import HttpOperator

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", 0.1))
FLASK_CONN_ID = 'flask_api'

default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=1),
}


def check_psi_threshold(**context):
    psi_response = context['ti'].xcom_pull(task_ids='get_psi')

    if not psi_response:
        context['ti'].xcom_push(key='psi_value', value=0)
        return 'no_action'

    psi_value = psi_response.get('psi', 0)
    context['ti'].xcom_push(key='psi_value', value=psi_value)

    print(f"🔍 PSI: {psi_value}, Threshold: {PSI_THRESHOLD}")

    return 'retrain' if psi_value > PSI_THRESHOLD else 'no_action'


def log_result(**context):
    psi_value = context['ti'].xcom_pull(task_ids='check_psi', key='psi_value')
    retrain_result = context['ti'].xcom_pull(task_ids='retrain')

    print(f"📊 Pipeline completed: PSI={psi_value}, Retrained={retrain_result is not None}")


with DAG(
        dag_id="drift_retrain_ab_pipeline",
        start_date=datetime(2026, 1, 1),  # ← Конкретная дата
        schedule="@daily",
        catchup=False,
        tags=["ml", "drift", "ab"],
        default_args=default_args,
) as dag:
    get_psi = HttpOperator(
        task_id='get_psi',
        http_conn_id=FLASK_CONN_ID,
        endpoint='/api/psi',
        method='GET',
        headers={'Content-Type': 'application/json'},
        response_filter=lambda response: response.json(),
        log_response=True,
        do_xcom_push=True,
    )

    check_psi = BranchPythonOperator(
        task_id='check_psi',
        python_callable=check_psi_threshold,
    )

    retrain = HttpOperator(
        task_id='retrain',
        http_conn_id=FLASK_CONN_ID,
        endpoint='/api/train',
        method='POST',
        headers={'Content-Type': 'application/json'},
        response_filter=lambda response: response.json(),
        log_response=True,
        do_xcom_push=True,
    )

    no_action = PythonOperator(
        task_id='no_action',
        python_callable=lambda: print("✅ No retrain needed"),
    )

    join = PythonOperator(
        task_id='log_result',
        python_callable=log_result,
        trigger_rule='none_failed_min_one_success',  # ← Выполнится даже если ветка skipped
    )

    get_psi >> check_psi >> [retrain, no_action] >> join