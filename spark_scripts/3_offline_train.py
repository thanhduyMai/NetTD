import argparse
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    # Truyền mốc thời gian chốt sổ để train (Ví dụ: Train bằng dữ liệu tính đến ngày 2025-08-31)
    parser.add_argument("--train_until_date", required=True, help="VD: 2025-08-31") 
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"Offline_Train_Baseline_until_{args.train_until_date}")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    silver_path = "s3a://silver/cell_kpi_clean/"
    model_output_path = "s3a://gold/cell_baseline_model/" # Nơi lưu "Mô hình" (Baseline)

    print(f" Bắt đầu OFFLINE TRAINING dữ liệu từ đầu đến ngày {args.train_until_date}...")

    # 1. Load dữ liệu lịch sử sạch
    df_silver = spark.read.format("delta").load(silver_path)
    df_history = df_silver.filter(F.col("date") <= args.train_until_date)

    
    # Tối ưu: Phân nhóm theo cả Trạm VÀ Khung giờ (vì ban đêm traffic luôn thấp hơn ban ngày)
    # 2. TRAINING (Tính Mean và StdDev cho Full 6 KPIs)
    df_baseline = df_history.groupBy("cell_name", "hour").agg(
        # Traffic
        F.avg("total_traffic_gb").alias("mean_total_traffic"),
        F.stddev("total_traffic_gb").alias("std_total_traffic"),
        F.avg("dl_traffic_gb").alias("mean_dl_traffic"),
        F.stddev("dl_traffic_gb").alias("std_dl_traffic"),
        F.avg("ul_traffic_gb").alias("mean_ul_traffic"),
        F.stddev("ul_traffic_gb").alias("std_ul_traffic"),
        # User Connected
        F.avg("max_rrc_connected_user").alias("mean_rrc_user"),
        F.stddev("max_rrc_connected_user").alias("std_rrc_user"),
        # PRB Used
        F.avg("dl_prb_used").alias("mean_dl_prb"),
        F.stddev("dl_prb_used").alias("std_dl_prb"),
        F.avg("ul_prb_used").alias("mean_ul_prb"),
        F.stddev("ul_prb_used").alias("std_ul_prb"),
        
        F.count("cell_name").alias("sample_count")
    )

    # Làm sạch: Fillna cho TẤT CẢ các cột std (đề phòng có 1 ngày dữ liệu nên std bị null)
    std_cols = ["std_total_traffic", "std_dl_traffic", "std_ul_traffic", "std_rrc_user", "std_dl_prb", "std_ul_prb"]
    df_baseline = df_baseline.fillna({c: 0.01 for c in std_cols})
    
    # Ép các std = 0 thành 0.01 để lúc chia không bị lỗi chia cho 0 (Divide by Zero)
    for c in std_cols:
        df_baseline = df_baseline.withColumn(c, F.when(F.col(c) == 0, 0.01).otherwise(F.col(c)))

    # Làm sạch các trạm thiếu dữ liệu (std = null hoặc 0)
    df_baseline = df_baseline.fillna({"std_traffic": 0.01, "std_prb": 0.01})
    df_baseline = df_baseline.withColumn("std_traffic", F.when(F.col("std_traffic") == 0, 0.01).otherwise(F.col("std_traffic")))
    df_baseline = df_baseline.withColumn("std_prb", F.when(F.col("std_prb") == 0, 0.01).otherwise(F.col("std_prb")))

    print("💾 Đang ghi Model Baseline xuống Gold layer...")
    # Ghi đè toàn bộ baseline mới nhất
    df_baseline.write.format("delta").mode("overwrite") \
        .option("mergeSchema", "true") \
        .save(model_output_path)

    print(" Offline Training hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()