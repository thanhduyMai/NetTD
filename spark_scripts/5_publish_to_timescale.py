import argparse
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_date_hour", required=True, help="VD: 2025-06-30-18") 
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"Publish_Timescale_{args.target_date_hour}")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        # Chú ý: Cần có thư viện JDBC của PostgreSQL để Spark nói chuyện được với TimescaleDB
        .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") 
        .getOrCreate()
    )

    target_date = args.target_date_hour[:10]
    target_hr = args.target_date_hour[-2:]
    
    silver_path = "s3a://silver/cell_kpi_clean/"
    anomaly_path = "s3a://gold/cell_kpi_anomalies/"

    # Cấu hình JDBC TimescaleDB (Bác thay đổi user/pass/IP cho đúng hệ thống nhé)
    jdbc_url = "jdbc:postgresql://timescale:5432/nettd_db"
    jdbc_properties = {
        "user": "postgres",
        "password": "postgres_password",
        "driver": "org.postgresql.Driver",
        "stringtype": "unspecified" # Quan trọng để map kiểu UUID/Timestamptz
    }

    print(f"Bắt đầu Publish dữ liệu ca {args.target_date_hour} lên TimescaleDB...")

    # ====================================================
    # 1. ĐẨY DỮ LIỆU DIM VÀ FACT KPI (TỪ SILVER)
    # ====================================================
    try:
        df_silver = spark.read.format("delta").load(silver_path) \
            .filter((F.col("date") == target_date) & (F.col("hour") == target_hr))
    except Exception as e:
        print(f"Không có dữ liệu Silver để đẩy: {e}")
        sys.exit(0)

    if df_silver.count() > 0:
        # A. Xử lý Dim Cell (Lấy các trạm duy nhất)
        df_dim = df_silver.select(
            "cell_name", "enb_name", "cell_type", "sector", "azimuth", "x", "y"
        ).dropDuplicates(["cell_name"])

        # Spark JDBC không hỗ trợ UPSERT (ON CONFLICT) chuẩn, nên mẹo ở đây là dùng mode "ignore" 
        # Nếu cell_name đã có trong DB thì nó bỏ qua, chưa có thì nó insert thêm (Rất an toàn)
        print("Đang đẩy dữ liệu Dimension (dim_cell)...")
        df_dim.write.jdbc(url=jdbc_url, table="dim_cell", mode="ignore", properties=jdbc_properties)

        # B. Xử lý Fact KPI (Ghép date và hour thành TIMESTAMPTZ)
        df_fact_kpi = df_silver.withColumn(
            "timestamp", 
            F.to_timestamp(F.concat_ws(" ", F.col("date"), F.col("hour")), "yyyy-MM-dd HH")
        ).select(
            "timestamp", "cell_name", "total_traffic_gb", "dl_traffic_gb", 
            "ul_traffic_gb", "max_rrc_connected_user", "dl_prb_used", "ul_prb_used"
        )

        print("Đang đẩy dữ liệu Fact (fact_cell_kpi)...")
        df_fact_kpi.write.jdbc(url=jdbc_url, table="fact_cell_kpi", mode="append", properties=jdbc_properties)

    # ====================================================
    # 2. ĐẨY DỮ LIỆU FACT ANOMALY (TỪ GOLD - NẾU CÓ)
    # ====================================================
    try:
        df_anomaly = spark.read.format("delta").load(anomaly_path) \
            .filter((F.col("date") == target_date) & (F.col("hour") == target_hr))
        
        if df_anomaly.count() > 0:
            df_fact_anomaly = df_anomaly.withColumn(
                "timestamp", 
                F.to_timestamp(F.concat_ws(" ", F.col("date"), F.col("hour")), "yyyy-MM-dd HH")
            ).select(
                "timestamp", "cell_name", "total_traffic_gb", "z_total_traffic",
                "max_rrc_connected_user", "z_rrc_user", "dl_prb_used", "z_dl_prb",
                "ul_prb_used", "z_ul_prb", "detect_time"
            )

            print("Đang đẩy dữ liệu Cảnh báo (fact_anomaly_events)...")
            df_fact_anomaly.write.jdbc(url=jdbc_url, table="fact_anomaly_events", mode="append", properties=jdbc_properties)
        else:
            print("Ca này không có Anomaly nào để đẩy.")
            
    except Exception as e:
        print("Chưa có thư mục Anomaly hoặc ca này không có lỗi.")

    print("Hoàn tất quá trình Publish lên TimescaleDB!")
    spark.stop()

if __name__ == "__main__":
    main()