from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import datetime
import os
import json


app = Flask(__name__)
CORS(app)

# =====================================================================
# TẢI DỮ LIỆU DỰ ĐOÁN TỪ CSV VÀO BỘ NHỚ RAM (CHỈ CHẠY 1 LẦN)
# ===================================================================== 
csv_path = os.path.join(os.getcwd(), "predictions_2026.csv") 

try:
    df_predictions = pd.read_csv(csv_path)
    df_predictions['Date'] = df_predictions['date'].astype(str)
    print(f"✅ Đã tải thành công dữ liệu dự báo từ: {csv_path}")
except Exception as e:
    print(f"⚠️ CẢNH BÁO: Không tìm thấy file CSV hoặc file bị lỗi. Lỗi: {e}")
    df_predictions = pd.DataFrame()

# TẢI ĐỘ TIN CẬY CỦA MÔ HÌNH VÀO RAM
eval_path = os.path.join(os.getcwd(), "model_metrics.json")
try:
    with open(eval_path, "r", encoding="utf-8") as f:
        model_metrics = json.load(f)
    print("✅ Đã tải thành công báo cáo độ tin cậy AI.")
except Exception as e:
    print(f"⚠️ Không tìm thấy file model_metrics.json: {e}")
    model_metrics = {}

# Tọa độ mặc định (Thành phố Hồ Chí Minh)
DEFAULT_LAT = 10.7626
DEFAULT_LON = 106.6602


# 1. THIẾT LẬP CACHE VÀ RETRY CHO API THÔNG THƯỜNG
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# 2. TỪ ĐIỂN ĐẦY ĐỦ CÁC TRƯỜNG DỮ LIỆU
ALL_FIELDS = {
    "daily": [
        "weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", 
        "apparent_temperature_min", "sunset", "sunrise", "daylight_duration", "sunshine_duration", 
        "uv_index_max", "uv_index_clear_sky_max", "precipitation_probability_max", "precipitation_hours", 
        "precipitation_sum", "snowfall_sum", "showers_sum", "rain_sum", "wind_speed_10m_max", 
        "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ],
    "hourly": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", 
        "precipitation_probability", "precipitation", "rain", "showers", "snowfall", "snow_depth", 
        "vapour_pressure_deficit", "et0_fao_evapotranspiration", "evapotranspiration", "visibility", 
        "cloud_cover_high", "cloud_cover_mid", "cloud_cover_low", "cloud_cover", "pressure_msl", 
        "weather_code", "surface_pressure", "wind_speed_10m", "wind_speed_80m", "wind_speed_120m", 
        "wind_speed_180m", "wind_direction_10m", "wind_direction_80m", "wind_direction_120m", 
        "wind_direction_180m", "wind_gusts_10m", "temperature_80m", "temperature_120m", "temperature_180m", 
        "soil_moisture_27_to_81cm", "soil_moisture_9_to_27cm", "soil_moisture_3_to_9cm", 
        "soil_moisture_1_to_3cm", "soil_temperature_54cm", "soil_moisture_0_to_1cm", 
        "soil_temperature_18cm", "soil_temperature_6cm", "soil_temperature_0cm"
    ],
    "minutely_15": [
        "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", 
        "shortwave_radiation", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", 
        "global_tilted_irradiance", "terrestrial_radiation", "terrestrial_radiation_instant", 
        "global_tilted_irradiance_instant", "direct_normal_irradiance_instant", "diffuse_radiation_instant", 
        "shortwave_radiation_instant", "sunshine_duration", "freezing_level_height", "snowfall", "rain", 
        "snowfall_height", "direct_radiation_instant", "wind_direction_80m", "wind_direction_10m", 
        "wind_speed_80m", "wind_speed_10m", "weather_code", "is_day", "lightning_potential", "visibility", 
        "wind_gusts_10m", "cape"
    ],
    "current": [
        "relative_humidity_2m", "temperature_2m", "is_day", "apparent_temperature", 
        "snowfall", "showers", "rain", "precipitation", "surface_pressure", "pressure_msl", 
        "cloud_cover", "weather_code", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"
    ]
}

@app.route('/api/weather', methods=['GET'])
def get_weather_data():
    try:
        api_type = request.args.get('api_type', 'forecast') 
        interval = request.args.get('interval', 'hourly')
        fields_param = request.args.get('fields') 
        lat = request.args.get('latitude', default=10.8231, type=float)
        lon = request.args.get('longitude', default=106.6297, type=float)
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        forecast_days = request.args.get('forecast_days')
        past_days = request.args.get('past_days')

        if interval not in ALL_FIELDS:
            return jsonify({"status": "error", "message": "Interval không hợp lệ"}), 400

        # Lấy tất cả các trường có sẵn cho interval đó
        all_fields_for_interval = ALL_FIELDS[interval]
        
        # SỬA LỖI: Ghi nhận yêu cầu của Client nhưng không xóa dữ liệu
        client_requested_fields = fields_param.split(',') if fields_param else all_fields_for_interval

        url_mapping = {
            'forecast': "https://api.open-meteo.com/v1/forecast",
            'archive': "https://archive-api.open-meteo.com/v1/archive",
            'air_quality': "https://air-quality-api.open-meteo.com/v1/air-quality"
        }
        url = url_mapping.get(api_type, url_mapping['forecast'])

        params = {
            "latitude": lat, "longitude": lon,
            "timezone": "Asia/Bangkok",
            interval: all_fields_for_interval # Luôn lấy 100% dữ liệu từ API
        }

        if start_date and end_date:
            params['start_date'], params['end_date'] = start_date, end_date
        if forecast_days:
            params['forecast_days'] = int(forecast_days)
        if past_days:
            params['past_days'] = int(past_days)

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        if interval == 'current':
            time_data = response.Current()
            current_time = pd.to_datetime(time_data.Time() + response.UtcOffsetSeconds(), unit="s", utc=True).strftime('%Y-%m-%d %H:%M:%S')
            current_data = {"date": current_time}
            for index, field in enumerate(all_fields_for_interval):
                current_data[field] = time_data.Variables(index).Value()
            result_json = [current_data]
        else:
            if interval == 'minutely_15': time_data = response.Minutely15()
            elif interval == 'hourly': time_data = response.Hourly()
            elif interval == 'daily': time_data = response.Daily()

            date_range = pd.date_range(
                start=pd.to_datetime(time_data.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
                end=pd.to_datetime(time_data.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=time_data.Interval()), inclusive="left"
            ).astype(str).tolist()

            data_dict = {"date": date_range}
            for index, field in enumerate(all_fields_for_interval):
                if field in ['sunrise', 'sunset']:
                    variable_data = time_data.Variables(index).ValuesInt64AsNumpy()
                else:
                    variable_data = time_data.Variables(index).ValuesAsNumpy()
                data_dict[field] = variable_data.tolist() if not isinstance(variable_data, int) else [None] * len(date_range)
            
            result_json = pd.DataFrame(data_dict).to_dict(orient='records')

        return jsonify({
            "status": "success",
            "meta": {
                "latitude": lat, 
                "longitude": lon, 
                "api_used": api_type, 
                "interval": interval,
                "client_requested_fields": client_requested_fields # Trả về danh sách client yêu cầu ở đây
            },
            "data": result_json # Trả về 100% dữ liệu gốc
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}), 500

# =====================================================================
# API DỰ ĐOÁN THỜI TIẾT TỪ MÔ HÌNH MACHINE LEARNING
# =====================================================================
@app.route('/api/predict_core', methods=['GET'])
def get_core_predictions():
    try:
        # Lấy ngày được yêu cầu từ URL (Ví dụ: /api/predict_core?date=2026-05-20)
        target_date_raw = request.args.get('date')

        if not target_date_raw:
            return jsonify({
                "status": "error", 
                "message": "Vui lòng cung cấp tham số 'date'"
            }), 400

        try:
            # pd.to_datetime sẽ biến "2026-9-5" thành "2026-09-05" một cách kỳ diệu
            target_date = pd.to_datetime(target_date_raw).strftime('%Y-%m-%d')
        except Exception:
            return jsonify({
                "status": "error", 
                "message": "Định dạng ngày không hợp lệ. Vui lòng nhập ngày thực tế."
            }), 400
        # -------------------------------------------------------------

        if df_predictions.empty:
            return jsonify({
                "status": "error", 
                "message": "Dữ liệu học máy chưa sẵn sàng. Vui lòng kiểm tra lại file CSV."
            }), 500

        # 2. Lọc ra dòng dữ liệu khớp với ngày đã được chuẩn hóa
        row_data = df_predictions[df_predictions['date'] == target_date]

        if row_data.empty:
            return jsonify({
                "status": "error", 
                "message": f"Không có kết quả dự báo cho ngày {target_date}."
            }), 404

        # Trích xuất dòng dữ liệu đầu tiên
        row = row_data.iloc[0]

        # Đóng gói dữ liệu thành JSON Nested
        response_data = {
            "status": "success",
            "meta": {
                "latitude": DEFAULT_LAT, 
                "longitude": DEFAULT_LON,
                "date_requested": target_date,
                "algorithms_used": ["RandomForest", "LinearRegression", "XGB_GBT"],
                "model_accuracy": model_metrics
            },
            "data": {
                "date": target_date,
                "temperature_2m_max": {
                    "RandomForest": row.get("temperature_2m_max_RandomForest"),
                    "LinearRegression": row.get("temperature_2m_max_LinearRegression"),
                    "XGB": row.get("temperature_2m_max_GBT")
                },
                "temperature_2m_min": {
                    "RandomForest": row.get("temperature_2m_min_RandomForest"),
                    "LinearRegression": row.get("temperature_2m_min_LinearRegression"),
                    "XGB": row.get("temperature_2m_min_GBT")
                },
                "precipitation_sum": {
                    "RandomForest": row.get("precipitation_sum_RandomForest"),
                    "LinearRegression": row.get("precipitation_sum_LinearRegression"),
                    "XGB": row.get("precipitation_sum_GBT")
                },
                "wind_speed_10m_max": {
                    "RandomForest": row.get("wind_speed_10m_max_RandomForest"),
                    "LinearRegression": row.get("wind_speed_10m_max_LinearRegression"),
                    "XGB": row.get("wind_speed_10m_max_GBT")
                }
            }
        }

        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi xử lý dự báo: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(debug=True, port=5000)