from airflow import DAG
import pendulum
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from datetime import datetime, timedelta
import os
import time
import paramiko

# Khai báo múi giờ local
local_tz = pendulum.timezone("Asia/Ho_Chi_Minh")

def fetch_sftp_to_minio_landing(execution_date, **kwargs):
    # 1. Dịch múi giờ UTC sang giờ VN trước khi ép kiểu
    local_time = execution_date.in_timezone('Asia/Ho_Chi_Minh')
    
    # Trích xuất thời gian chạy theo THÁNG (YYYYMM)
    year_month = local_time.strftime('%Y%m')    # Đảm bảo chắc chắn ra 202506, 202507...
    file_name = f"{year_month}.csv"                 # VD: 202506.csv
    
    # 2. Cấu hình kết nối
    conn_id = 'sftp_server' 
    print(f" Đang đọc cấu hình '{conn_id}' từ Airflow...")
    try:
        conn = BaseHook.get_connection(conn_id)
    except Exception as e:
        raise ValueError(f"Không tìm thấy Connection '{conn_id}': {e}")

    host = conn.host
    port = conn.port if conn.port else 22
    user = conn.login
    password = conn.password

    # 3. Đường dẫn mới cho thư mục Cell KPI
    remote_file = f"/u01/vdt-data-de/cell-kpi/{file_name}"
    local_file = f"/tmp/cell_kpi_{file_name}"
    minio_bucket = Variable.get("minio_landing_bucket", default_var="landingzone")

    print(f" Bắt đầu kéo dữ liệu từ {user}@{host}:{port}...")
    print(f"  -> File Nguồn: {remote_file}")
    
    # ==========================================
    # PHẦN 1: TẢI TỪ SFTP VỀ WORKER
    # ==========================================
    download_time = 0.0
    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=user, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        
        # Kiểm tra file có tồn tại trên server không
        try:
            sftp.stat(remote_file)
        except IOError:
            raise FileNotFoundError(f"File không tồn tại trên SFTP server: {remote_file}")

        print(f" Đang tải {file_name}... ", end='', flush=True)
        start_time = time.time()
        
        # Lấy file về /tmp/ của Airflow Worker
        sftp.get(remote_file, local_file)
        
        download_time = time.time() - start_time
        print(f" ({download_time:.3f}s)")

        sftp.close()
        transport.close()
    except Exception as e:
        raise RuntimeError(f" LỖI TRONG QUÁ TRÌNH TẢI SFTP: {e}")

    # ==========================================
    # PHẦN 2: ĐẨY TỪ WORKER LÊN MINIO LAKEHOUSE
    # ==========================================
    print(f" Đang upload file lên MinIO: s3a://{minio_bucket}/{file_name}...")
    s3_start_time = time.time()
    
    s3_hook = S3Hook(aws_conn_id='minio_default')
    s3_hook.load_file(
        filename=local_file,
        key=file_name,
        bucket_name=minio_bucket,
        replace=True
    )
    
    upload_time = time.time() - s3_start_time
    
    # Xóa file tạm tại worker
    if os.path.exists(local_file):
        os.remove(local_file)

    # ==========================================
    # BÁO CÁO HIỆU NĂNG
    # ==========================================
    print('\n=======================================================')
    print(f' BÁO CÁO HIỆU NĂNG TẢI DỮ LIỆU THÁNG {year_month}')
    print('=======================================================')
    print(f' Thời gian tải (SFTP -> Worker)  : {download_time:.3f} giây')
    print(f' Thời gian lưu (Worker -> MinIO) : {upload_time:.3f} giây')
    print(f' Tổng thời gian Ingestion        : {download_time + upload_time:.3f} giây')
    print(f' Dữ liệu đích sẵn sàng tại       : s3a://{minio_bucket}/{file_name}')
    print('=======================================================\n')

    return f"s3a://{minio_bucket}/{file_name}"


# 1. Cấu hình mặc định DAG
default_args = {
    'owner': 'data_engineer_team',
    'depends_on_past': False,
    # Gắn múi giờ vào để Airflow UI hiển thị chuẩn giờ VN và start đúng từ tháng 6
    'start_date': datetime(2025, 6, 1, 0, 0, tzinfo=local_tz),  
    'end_date': datetime(2025, 9, 30, 23, 59, tzinfo=local_tz), 
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# 2. Khởi tạo DAG
with DAG(
    'cell_kpi_spatial_analytics_pipeline',
    default_args=default_args,
    description='Pipeline Cell KPI Lakehouse với SFTP Paramiko',
    schedule_interval='@monthly',  
    catchup=True,                  
    max_active_runs=1, 
    tags=['cell_kpi', 'star_schema', 'sftp']
) as dag:

    common_args = [
        '--date', '{{ ds }}'
    ]
    
    # Sửa lại Jinja Template ép múi giờ VN để truyền đúng file YYYYMM.csv cho Spark
    ingest_args = [
        '--type', 'cell_kpi',
        '--date', '{{ ds }}',
        '--input_path', "s3a://" + Variable.get("minio_landing_bucket", "landingzone") + "/{{ execution_date.in_timezone('Asia/Ho_Chi_Minh').strftime('%Y%m') }}.csv"
    ]

    # TASK 0: FETCH SFTP
    task_fetch_sftp = PythonOperator(
        task_id='0_fetch_sftp_to_minio',
        python_callable=fetch_sftp_to_minio_landing,
    )


    # 3. KẾT NỐI ĐƯỜNG ỐNG
    task_fetch_sftp