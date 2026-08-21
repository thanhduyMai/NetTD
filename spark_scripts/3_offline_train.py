import argparse
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_until_date", required=True, help="VD: 2025-08-31") 
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"Offline_Train_Baseline_until_{args.train_until_date}")
        # Ép cứng nó gọi vào container tên là nettd_minio hoặc minio ở cổng 9000
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.path.style.access", "true") # BẮT BUỘC ĐỂ MINIO HIỂU ĐÚNG ĐƯỜNG DẪN
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    silver_path = "s3a://silver/cell_kpi_clean/"
    model_output_path = "s3a://gold/cell_baseline_model/"

    print(f"Bắt đầu OFFLINE TRAINING dữ liệu từ đầu đến ngày {args.train_until_date}...")

    # 1. Load dữ liệu lịch sử sạch
    try:
        df_silver = spark.read.format("delta").load(silver_path)
    except Exception as e:
        print(f"Chưa có dữ liệu ở Silver: {e}. Skip Train.")
        sys.exit(0)

    df_history = df_silver.filter(F.col("date") <= args.train_until_date)

    # ---------------------------------------------------------
    # 2. CHỐT CHẶN BẢO VỆ: ĐẢM BẢO ĐỦ MẪU ĐỂ TÍNH STDDEV
    # ---------------------------------------------------------
    # Ít nhất phải có 2 ngày dữ liệu mới có 2 điểm dữ liệu ở cùng 1 khung giờ
    days_collected = df_history.select("date").distinct().count()
    if days_collected < 2:
        print(f"Dữ liệu mới có {days_collected} ngày. Chưa đủ số mẫu (N>=2) để tính Độ lệch chuẩn (StdDev).")
        print("Bỏ qua Offline Training, thông luồng chạy tiếp!")
        sys.exit(0) # Báo SUCCESS cho Airflow

    # 3. TRAINING (Tính Mean và StdDev cho Full 6 KPIs)
    df_baseline = df_history.groupBy("cell_name", "hour").agg(
        F.avg("total_traffic_gb").alias("mean_total_traffic"),
        F.stddev("total_traffic_gb").alias("std_total_traffic"),
        F.avg("dl_traffic_gb").alias("mean_dl_traffic"),
        F.stddev("dl_traffic_gb").alias("std_dl_traffic"),
        F.avg("ul_traffic_gb").alias("mean_ul_traffic"),
        F.stddev("ul_traffic_gb").alias("std_ul_traffic"),
        F.avg("max_rrc_connected_user").alias("mean_rrc_user"),
        F.stddev("max_rrc_connected_user").alias("std_rrc_user"),
        F.avg("dl_prb_used").alias("mean_dl_prb"),
        F.stddev("dl_prb_used").alias("std_dl_prb"),
        F.avg("ul_prb_used").alias("mean_ul_prb"),
        F.stddev("ul_prb_used").alias("std_ul_prb"),
        F.count("cell_name").alias("sample_count")
    )

    std_cols = ["std_total_traffic", "std_dl_traffic", "std_ul_traffic", "std_rrc_user", "std_dl_prb", "std_ul_prb"]
    df_baseline = df_baseline.fillna({c: 0.01 for c in std_cols})
    
    for c in std_cols:
        df_baseline = df_baseline.withColumn(c, F.when(F.col(c) == 0, 0.01).otherwise(F.col(c)))

    print("Đang ghi Model Baseline xuống Gold layer...")
    df_baseline.write.format("delta").mode("overwrite") \
        .option("mergeSchema", "true") \
        .save(model_output_path)

    print("Offline Training hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()