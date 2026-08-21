from pyspark.sql import SparkSession

def main():
    # Khởi tạo Spark kết nối với MinIO
    spark = (
        SparkSession.builder.appName("Test_Schema_Cell_KPI")
        .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
        .config("spark.hadoop.fs.s3a.access.key", "admin")
        .config("spark.hadoop.fs.s3a.secret.key", "password")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )

    # Đường dẫn tới file ở MinIO (giả sử file tháng 6)
    # Nếu bác đang test file tháng khác thì đổi số 202506 nhé
    input_path = "s3a://landingzone/202506.csv"
    
    print(f"\n ĐANG ĐỌC FILE : {input_path}")
    
    try:
        # Đọc file CSV, nhớ bật inferSchema để nó tự đoán kiểu dữ liệu (String, Integer, Timestamp...)
        df = spark.read.csv(input_path, header=True, inferSchema=True)
        
        print("\n" + "="*50)
        print(" THÔNG TIN SCHEMA CỦA DỮ LIỆU ")
        print("="*50)
        df.printSchema()
        
        print("\n" + "="*50)
        print(" 5 DÒNG DỮ LIỆU ĐẦU TIÊN ĐỂ KIỂM CHỨNG ")
        print("="*50)
        df.show(5, truncate=False)
        
        print(f"\n TỔNG SỐ DÒNG TRONG FILE NÀY: {df.count():,} dòng")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra khi đọc file: {e}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    main()