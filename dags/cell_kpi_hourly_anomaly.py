from airflow import DAG
import pendulum
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from datetime import datetime, timedelta
import os

local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

def check_and_fetch_landing(execution_date, **kwargs):
    local_time = execution_date.in_timezone('Asia/Ho_Chi_Minh')
    year_month = local_time.strftime('%Y%m')    
    file_name = f"{year_month}.csv"                 
    minio_bucket = Variable.get("minio_landing_bucket", default_var="landingzone")

    s3_hook = S3Hook(aws_conn_id='minio_default')
    # Nếu file đã có, bỏ qua việc kết nối SFTP
    if s3_hook.check_for_key(key=file_name, bucket_name=minio_bucket):
        print(f" File {file_name} đã sẵn sàng trên Landing Zone.")
        return f"s3a://{minio_bucket}/{file_name}"
        
    raise FileNotFoundError(f"File {file_name} chưa tồn tại trên MinIO. Hãy chạy DAG Monthly trước!")

default_args = {
    'owner': 'data_engineer_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 1, 0, 0, tzinfo=local_tz),  
    'end_date': datetime(2025, 9, 30, 23, 59, tzinfo=local_tz), 
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'cell_kpi_hourly_anomaly_pipeline',
    default_args=default_args,
    description='Pipeline chạy theo giờ và phát hiện bất thường',
    schedule_interval='@hourly',  
    catchup=True,                  
    max_active_runs=2,
    tags=['cell_kpi', 'hourly', 'anomaly']
) as dag:

    input_csv_path = "s3a://" + Variable.get("minio_landing_bucket", "landingzone") + "/{{ execution_date.in_timezone('Asia/Ho_Chi_Minh').strftime('%Y%m') }}.csv"
    
    # Định dạng giờ cho Ingest, QC, Inference, Publish
    target_date_hour = "{{ execution_date.in_timezone('Asia/Ho_Chi_Minh').strftime('%Y-%m-%d-%H') }}"
    
    # Định dạng ngày (Cắt bớt giờ) dành riêng cho task Offline Train
    target_date = "{{ execution_date.in_timezone('Asia/Ho_Chi_Minh').strftime('%Y-%m-%d') }}"

    task_check_file = PythonOperator(
        task_id='0_check_landing_file',
        python_callable=check_and_fetch_landing,
    )

    task_ingest_hourly = SparkSubmitOperator(
        task_id='1_ingest_hourly_bronze',
        application='/opt/spark_scripts/1_ingest_hourly.py', 
        conn_id='spark-master', 
        application_args=['--input_path', input_csv_path, '--target_date_hour', target_date_hour] 
    )
    
    task_qc_silver = SparkSubmitOperator(
        task_id='2_qc_hourly_silver',
        application='/opt/spark_scripts/2_qc_hourly_silver.py',
        conn_id='spark-master',
        application_args=['--target_date_hour', target_date_hour]
    )

    # THÊM TASK 3: TÍNH TOÁN BASELINE MODEL
    task_offline_train = SparkSubmitOperator(
        task_id='3_offline_train_baseline',
        application='/opt/spark_scripts/3_offline_train_anomaly.py',
        conn_id='spark-master',
        application_args=['--train_until_date', target_date]
    )

    # THÊM TASK 4: SUY LUẬN ANOMALY
    task_online_inference = SparkSubmitOperator(
        task_id='4_online_inference_anomaly',
        application='/opt/spark_scripts/4_online_inference_anomaly.py',
        conn_id='spark-master',
        application_args=['--target_date_hour', target_date_hour]
    )

    # THÊM TASK 5: ĐẨY DỮ LIỆU LÊN TIMESCALEDB
    task_publish_timescale = SparkSubmitOperator(
        task_id='5_publish_to_timescale',
        application='/opt/spark_scripts/5_publish_to_timescale.py',
        conn_id='spark-master',
        conf={'spark.jars.packages': 'org.postgresql:postgresql:42.6.0'},
        application_args=['--target_date_hour', target_date_hour]
    )

    # LIÊN KẾT LUỒNG CHẠY TUẦN TỰ
    task_check_file >> task_ingest_hourly >> task_qc_silver >> task_offline_train >> task_online_inference >> task_publish_timescale