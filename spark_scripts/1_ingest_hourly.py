import argparse
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Đường dẫn file CSV Tháng ở Landing Zone")
    parser.add_argument("--target_date_hour", required=True, help="Giờ cần bóc. VD: 2025-06-30-18")
    args = parser.parse_args()

    # Khởi tạo Spark Session với cấu hình MinIO
    spark = (
        SparkSession.builder.appName(f"Ingest_Bronze_{args.target_date_hour}")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )

    output_path = "s3a://bronze/cell_kpi_raw/"
    
    print(f"Đang nạp {args.input_path} và trích xuất dữ liệu ca: {args.target_date_hour}...")
    
    # 1. Đọc file CSV (Bắt buộc dùng sep="|" để nó cắt cột)
    raw_df = spark.read.csv(args.input_path, header=True, inferSchema=True, sep="|")
    
    # MẸO BULLETPROOF: Lột sạch dấu ngoặc kép (") ở tất cả tên cột do CSV sinh ra
    for c in raw_df.columns:
        raw_df = raw_df.withColumnRenamed(c, c.replace('"', ''))
        
    # Lột tiếp dấu ngoặc kép ở dữ liệu của cột date_hour (Ví dụ từ "2025-06-30-18" thành 2025-06-30-18)
    clean_df = raw_df.withColumn("date_hour", F.regexp_replace(F.col("date_hour"), '"', ''))

    # 2. Lọc đúng giờ
    df_hourly = clean_df.filter(F.col("date_hour") == args.target_date_hour)
    
    if df_hourly.count() == 0:
        print(f"Không có dữ liệu cho ca {args.target_date_hour}. Dừng luồng an toàn.")
        sys.exit(0)

    # 3. Tạo cột date và hour để chia thư mục (Partition)
    df_partitioned = df_hourly \
        .withColumn("date", F.substring(F.col("date_hour"), 1, 10)) \
        .withColumn("hour", F.substring(F.col("date_hour"), 12, 2))
    
    # Lấy ra chuỗi ngày và giờ để làm điều kiện ghi đè (replaceWhere)
    target_date = args.target_date_hour[:10]
    target_hr = args.target_date_hour[-2:]
    
    print(f"Đang ghi {df_partitioned.count()} dòng xuống Bronze layer (Partition: {target_date} / {target_hr})...")
    
    # 4. Ghi xuống Delta Lake
    df_partitioned.write.format("delta") \
        .mode("overwrite") \
        .option("mergeSchema", "true") \
        .option("replaceWhere", f"date = '{target_date}' AND hour = '{target_hr}'") \
        .partitionBy("date", "hour") \
        .save(output_path)
        
    print("Ingest Bronze Hourly hoàn tất!")
    spark.stop()

if __name__ == "__main__":
    main()