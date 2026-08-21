import argparse
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_date_hour", required=True, help="VD: 2025-09-01-08") 
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"Online_Inference_{args.target_date_hour}")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    target_date = args.target_date_hour[:10]
    target_hr = args.target_date_hour[-2:]
    
    silver_path = "s3a://silver/cell_kpi_clean/"
    model_path = "s3a://gold/cell_baseline_model/" 
    anomaly_output_path = "s3a://gold/cell_kpi_anomalies/"

    print(f"Bắt đầu tiến trình ONLINE INFERENCE cho ca {args.target_date_hour}...")

    # 1. Đếm số ngày dữ liệu đã tích lũy trong Silver
    try:
        df_history = spark.read.format("delta").load(silver_path)
    except Exception as e:
         print(f"Chưa có dữ liệu ở Silver. Vui lòng chạy luồng Ingest trước! Lỗi: {e}")
         sys.exit(0)
         
    days_collected = df_history.select("date").distinct().count()
    
    # LUẬT "NGỦ ĐÔNG" DYNAMIC: Chờ đủ 90 ngày
    if days_collected < 90:
        print(f"Mới thu thập được {days_collected}/90 ngày. Đang tích lũy, bỏ qua Online Inference.")
        sys.exit(0)

    # 2. Load dữ liệu hiện tại của đúng ca giờ này
    df_current = df_history.filter((F.col("date") == target_date) & (F.col("hour") == target_hr))

    if df_current.count() == 0:
        print("Không có dữ liệu hiện tại để suy luận.")
        sys.exit(0)

    # 3. Load Model Baseline
    try:
        df_baseline = spark.read.format("delta").load(model_path)
    except Exception as e:
        print(f"Chưa có Model Baseline. Hãy chạy luồng Offline Training trước! Lỗi: {e}")
        sys.exit(0)

    # 4. ONLINE INFERENCE: Tính Z-Score cho 6 KPIs
    df_infer = df_current.join(df_baseline, on=["cell_name", "hour"], how="left")

    # Bỏ qua các trạm chưa từng xuất hiện trong lịch sử
    df_infer = df_infer.filter(F.col("mean_total_traffic").isNotNull())

    df_scored = df_infer.withColumn(
        "z_total_traffic", F.abs((F.col("total_traffic_gb") - F.col("mean_total_traffic")) / F.col("std_total_traffic"))
    ).withColumn(
        "z_dl_traffic", F.abs((F.col("dl_traffic_gb") - F.col("mean_dl_traffic")) / F.col("std_dl_traffic"))
    ).withColumn(
        "z_ul_traffic", F.abs((F.col("ul_traffic_gb") - F.col("mean_ul_traffic")) / F.col("std_ul_traffic"))
    ).withColumn(
        "z_rrc_user", F.abs((F.col("max_rrc_connected_user") - F.col("mean_rrc_user")) / F.col("std_rrc_user"))
    ).withColumn(
        "z_dl_prb", F.abs((F.col("dl_prb_used") - F.col("mean_dl_prb")) / F.col("std_dl_prb"))
    ).withColumn(
        "z_ul_prb", F.abs((F.col("ul_prb_used") - F.col("mean_ul_prb")) / F.col("std_ul_prb"))
    )

    # 5. DETECT THRESHOLD: Báo động nếu bất kỳ KPI nào có Z-Score > 3
    anomaly_condition = (
        (F.col("z_total_traffic") > 3.0) |
        (F.col("z_rrc_user") > 3.0) |
        (F.col("z_dl_prb") > 3.0) |
        (F.col("z_ul_prb") > 3.0) |
        (F.col("z_dl_traffic") > 3.0) |
        (F.col("z_ul_traffic") > 3.0)
    )

    df_anomalies = df_scored.filter(anomaly_condition)

    anomaly_count = df_anomalies.count()
    print(f"Suy luận xong. Phát hiện {anomaly_count} trạm bất thường!")

    # 6. LƯU XUỐNG GOLD
    if anomaly_count > 0:
        df_anomalies_to_save = df_anomalies.withColumn("detect_time", F.current_timestamp())
        
        cols_to_save = [
            "cell_name", "date", "hour", "x", "y", 
            "total_traffic_gb", "mean_total_traffic", "z_total_traffic",
            "max_rrc_connected_user", "mean_rrc_user", "z_rrc_user",
            "dl_prb_used", "z_dl_prb", "ul_prb_used", "z_ul_prb",
            "detect_time"
        ]
        
        df_anomalies_to_save.select(*cols_to_save).write.format("delta") \
            .mode("overwrite") \
            .option("mergeSchema", "true") \
            .option("replaceWhere", f"date = '{target_date}' AND hour = '{target_hr}'") \
            .partitionBy("date", "hour") \
            .save(anomaly_output_path)
        print("Đã bắn cảnh báo xuống thư mục Anomalies.")

    spark.stop()

if __name__ == "__main__":
    main()