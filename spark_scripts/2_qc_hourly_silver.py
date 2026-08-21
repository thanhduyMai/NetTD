import argparse
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_date_hour", required=True, help="VD: 2025-06-30-18")
    args = parser.parse_args()

    # Khởi tạo Spark Session
    spark = (
        SparkSession.builder.appName(f"QC_Silver_{args.target_date_hour}")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    bronze_path = "s3a://bronze/cell_kpi_raw/"
    silver_clean_path = "s3a://silver/cell_kpi_clean/"
    silver_dirty_path = "s3a://silver/cell_kpi_dirty/"

    target_date = args.target_date_hour[:10]
    target_hr = args.target_date_hour[-2:]

    print(f"Bắt đầu Quality Check (QC) cho ca: {args.target_date_hour}")

    try:
        # Chỉ load đúng Partition của giờ hiện tại từ tầng Bronze
        df_target = spark.read.format("delta").load(bronze_path) \
            .filter((F.col("date") == target_date) & (F.col("hour") == target_hr))
    except Exception as e:
        print(f"Lỗi đọc Bronze (Có thể chưa có dữ liệu): {e}")
        sys.exit(0)

    total_records = df_target.count()
    if total_records == 0:
        print("Phân vùng trống. Không có dữ liệu để QC.")
        sys.exit(0)

    # ==========================================
    # ĐỊNH NGHĨA LUẬT LÀM SẠCH (DATA QUALITY RULES)
    # Dựa vào schema Cell KPI bác cung cấp
    # ==========================================
    clean_condition = (
        F.col("cell_name").isNotNull() &                # Tên trạm không được rỗng
        F.col("x").isNotNull() &                        # Tọa độ X (Lat) không được rỗng
        F.col("y").isNotNull() &                        # Tọa độ Y (Lng) không được rỗng
        (F.col("total_traffic_gb") >= 0) &              # Dung lượng phải >= 0
        (F.col("dl_prb_used") >= 0)                     # Tài nguyên vô tuyến >= 0
    )

    df_clean = df_target.filter(clean_condition)
    df_dirty = df_target.filter(~clean_condition)

    clean_count = df_clean.count()
    dirty_count = df_dirty.count()
    
    print(f"Tổng số bản ghi: {total_records} | Sạch: {clean_count} | Lỗi: {dirty_count}")

    # Ghi dữ liệu SẠCH xuống Silver
    if clean_count > 0:
        print(f"Ghi dữ liệu SẠCH xuống: {silver_clean_path}")
        df_clean.write.format("delta").mode("overwrite") \
            .option("mergeSchema", "true") \
            .option("replaceWhere", f"date = '{target_date}' AND hour = '{target_hr}'") \
            .partitionBy("date", "hour") \
            .save(silver_clean_path)

    # Ghi dữ liệu LỖI ra khu vực Dirty để sau này monitor
    if dirty_count > 0:
        print(f"Ghi dữ liệu LỖI ra: {silver_dirty_path}")
        df_dirty.write.format("delta").mode("overwrite") \
            .option("mergeSchema", "true") \
            .option("replaceWhere", f"date = '{target_date}' AND hour = '{target_hr}'") \
            .partitionBy("date", "hour") \
            .save(silver_dirty_path)

    print("Quy trình QC Silver hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()