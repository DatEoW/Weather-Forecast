# Hãy chắc chắn bạn đã thêm dòng import này ở đầu file ml_forecaster.py:
from pyspark.ml.evaluation import RegressionEvaluator
import os
import sys
import pandas as pd
import datetime
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, dayofyear, lag, to_date
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GBTRegressor

# 1. ÉP PYSPARK CHẠY TRÊN MẠNG NỘI BỘ
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. CẤU HÌNH SPARK (TỐI ƯU CHO WINDOWS)
# Tăng cường sức mạnh cho Spark trên Windows
spark = (SparkSession.builder
    .appName("Weather_Core_Ensemble")
    .master("local[2]")
    .config("spark.driver.memory", "16g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .getOrCreate())

spark.sparkContext.setLogLevel("ERROR")

# các dữ liệu cần dự đoán
CORE_FIELDS = [
    "temperature_2m_max", 
    "temperature_2m_min", 
    "precipitation_sum",
    "wind_speed_10m_max"
    ]

def fetch_core_historical_data(lat, lon):
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=365)

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": CORE_FIELDS, "timezone": "Asia/Bangkok"
    }

    responses = openmeteo.weather_api(url, params=params)
    daily = responses[0].Daily()
    date_range = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left"
    )
    
    data_dict = {"date": date_range.astype(str)}
    for index, field in enumerate(CORE_FIELDS):
        data_dict[field] = daily.Variables(index).ValuesAsNumpy().tolist()

    df = pd.DataFrame(data_dict).dropna()
    if df.empty: raise ValueError("Dữ liệu rỗng.")
    return df

def predict_core_weather(lat, lon, days_to_predict):
    pdf = fetch_core_historical_data(lat, lon)
    df = spark.createDataFrame(pdf).withColumn("date", to_date(col("date")))
    
    current_data = df.orderBy(col("date").desc()).first()
    current_date = current_data["date"]
    
    results_by_date = {}
    for i in range(1, days_to_predict + 1):
        future_date = (current_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        results_by_date[future_date] = {}

    for field in CORE_FIELDS:
        windowSpec = Window.orderBy("date")
        df_field = df.withColumn("lag_1", lag(field, 1).over(windowSpec)) \
                     .withColumn("lag_2", lag(field, 2).over(windowSpec)) \
                     .withColumn("month", month(col("date"))) \
                     .withColumn("day_of_year", dayofyear(col("date"))).dropna()

        assembler = VectorAssembler(inputCols=["month", "day_of_year", "lag_1", "lag_2"], outputCol="features")
        ml_data = assembler.transform(df_field)

        # Huấn luyện (Giữ nguyên 3 thuật toán)
        rf_model = RandomForestRegressor(featuresCol="features", labelCol=field, numTrees=5).fit(ml_data)
        lr_model = LinearRegression(featuresCol="features", labelCol=field).fit(ml_data)
        gbt_model = GBTRegressor(featuresCol="features", labelCol=field, maxIter=5).fit(ml_data)

        # Giá trị khởi tạo
        l_rf = {"v1": float(current_data[field]), "v2": float(df_field.orderBy(col("date").desc()).first()["lag_1"])}
        l_lr = {"v1": l_rf["v1"], "v2": l_rf["v2"]}
        l_gbt = {"v1": l_rf["v1"], "v2": l_rf["v2"]}

        for i in range(1, days_to_predict + 1):
            future_dt = current_date + datetime.timedelta(days=i)
            m, d_y = future_dt.month, future_dt.timetuple().tm_yday
            
            # --- TỐI ƯU: Dự đoán RF ---
            row_rf = spark.createDataFrame([(m, d_y, l_rf["v1"], l_rf["v2"])], ["month", "day_of_year", "lag_1", "lag_2"])
            p_rf = rf_model.transform(assembler.transform(row_rf)).select("prediction").first()[0]
            
            # --- TỐI ƯU: Dự đoán LR ---
            row_lr = spark.createDataFrame([(m, d_y, l_lr["v1"], l_lr["v2"])], ["month", "day_of_year", "lag_1", "lag_2"])
            p_lr = lr_model.transform(assembler.transform(row_lr)).select("prediction").first()[0]

            # --- TỐI ƯU: Dự đoán GBT ---
            row_gbt = spark.createDataFrame([(m, d_y, l_gbt["v1"], l_gbt["v2"])], ["month", "day_of_year", "lag_1", "lag_2"])
            p_gbt = gbt_model.transform(assembler.transform(row_gbt)).select("prediction").first()[0]

            results_by_date[future_dt.strftime("%Y-%m-%d")][field] = {
                "RandomForest": round(float(p_rf), 2),
                "LinearRegression": round(float(p_lr), 2),
                "XGBoost_GBT": round(float(p_gbt), 2)
            }
            # Cập nhật trễ
            l_rf["v2"], l_rf["v1"] = l_rf["v1"], p_rf
            l_lr["v2"], l_lr["v1"] = l_lr["v1"], p_lr
            l_gbt["v2"], l_gbt["v1"] = l_gbt["v1"], p_gbt

    return [{"date": k, **v} for k, v in results_by_date.items()]



# Dán hàm này vào DƯỚI CÙNG của file ml_forecaster.py
def evaluate_models(lat, lon):
    """
    Hàm chuyên dụng để chấm điểm mô hình. Không ảnh hưởng đến dự báo Web.
    Sẽ đánh giá cả 3 thuật toán dựa trên tập Train (70%) và Test (30%).
    """
    pdf = fetch_core_historical_data(lat, lon)
    df = spark.createDataFrame(pdf).withColumn("date", to_date(col("date")))
    
    evaluation_results = {}

    for field in CORE_FIELDS:
        windowSpec = Window.orderBy("date")
        df_field = df.withColumn("lag_1", lag(field, 1).over(windowSpec)) \
                     .withColumn("lag_2", lag(field, 2).over(windowSpec)) \
                     .withColumn("month", month(col("date"))) \
                     .withColumn("day_of_year", dayofyear(col("date"))).dropna()

        assembler = VectorAssembler(inputCols=["month", "day_of_year", "lag_1", "lag_2"], outputCol="features")
        ml_data = assembler.transform(df_field)

        # 1. Chia dữ liệu
        train_data, test_data = ml_data.randomSplit([0.7, 0.3], seed=42)

        # 2. Huấn luyện với thông số chuẩn (nhiều Tree hơn để đánh giá chính xác)
        rf_model = RandomForestRegressor(featuresCol="features", labelCol=field, numTrees=100).fit(train_data)
        lr_model = LinearRegression(featuresCol="features", labelCol=field).fit(train_data)
        gbt_model = GBTRegressor(featuresCol="features", labelCol=field, maxIter=50).fit(train_data)

        # 3. Dự đoán trên tập Test
        rf_preds = rf_model.transform(test_data)
        lr_preds = lr_model.transform(test_data)
        gbt_preds = gbt_model.transform(test_data)

        # 4. Khởi tạo công cụ chấm điểm
        eval_rmse = RegressionEvaluator(labelCol=field, predictionCol="prediction", metricName="rmse")
        eval_mae = RegressionEvaluator(labelCol=field, predictionCol="prediction", metricName="mae")
        eval_r2 = RegressionEvaluator(labelCol=field, predictionCol="prediction", metricName="r2")
        eval_mse = RegressionEvaluator(labelCol=field, predictionCol="prediction", metricName="mse")

        # 5. Đóng gói kết quả thành JSON
        evaluation_results[field] = {
            "RandomForest": {
                "RMSE": round(eval_rmse.evaluate(rf_preds), 4),
                "MAE": round(eval_mae.evaluate(rf_preds), 4),
                "R2": round(eval_r2.evaluate(rf_preds), 4),
                "MSE": round(eval_mse.evaluate(rf_preds), 4)
            },
            "LinearRegression": {
                "RMSE": round(eval_rmse.evaluate(lr_preds), 4),
                "MAE": round(eval_mae.evaluate(lr_preds), 4),
                "R2": round(eval_r2.evaluate(lr_preds), 4),
                "MSE": round(eval_mse.evaluate(lr_preds), 4)
            },
            "XGBoost_GBT": {
                "RMSE": round(eval_rmse.evaluate(gbt_preds), 4),
                "MAE": round(eval_mae.evaluate(gbt_preds), 4),
                "R2": round(eval_r2.evaluate(gbt_preds), 4),
                "MSE": round(eval_mse.evaluate(gbt_preds), 4)
            }
        }

    return evaluation_results