from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import datetime

# Import logic AI từ file ml_forecaster.py
from ml_forecaster import predict_core_weather,evaluate_models

app = Flask(__name__)
CORS(app)

# =====================================================================
# KHỞI ĐỘNG AI VÀ TÍNH TOÁN TRƯỚC (PRE-COMPUTATION)
# ===================================================================== 
DEFAULT_LAT = 16.1667
DEFAULT_LON = 107.8333
MAX_PREDICT_DAYS = 30 
PRECALCULATED_DATA = []

print("\n" + "="*60)
print("🚀 HỆ THỐNG ĐANG KHỞI ĐỘNG...")
try:
    PRECALCULATED_DATA = predict_core_weather(DEFAULT_LAT, DEFAULT_LON, MAX_PREDICT_DAYS)
    print("✅ HUẤN LUYỆN HOÀN TẤT! AI đã sẵn sàng phục vụ.")
    print("="*60 + "\n")
except Exception as e:
    print(f"❌ Lỗi khởi động AI: {e}")
    PRECALCULATED_DATA = []

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

@app.route('/api/predict_core', methods=['GET'])
def get_core_predictions():
    try:
        days_requested = int(request.args.get('days', 7))
        if days_requested <= 0 or days_requested > MAX_PREDICT_DAYS:
             return jsonify({"status": "error", "message": f"Hỗ trợ dự báo tối đa {MAX_PREDICT_DAYS} ngày."}), 400
        if not PRECALCULATED_DATA:
            return jsonify({"status": "error", "message": "Mô hình AI chưa sẵn sàng."}), 500

        prediction_result = PRECALCULATED_DATA[:days_requested]

        return jsonify({
            "status": "success",
            "meta": {
                "latitude": DEFAULT_LAT, "longitude": DEFAULT_LON,
                "days_predicted": days_requested,
                "algorithms": ["RandomForest", "LinearRegression", "XGBoost_GBT"]
            },
            "data": prediction_result
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# 2. Dán Endpoint mới này vào dưới cùng, ngay trên khối if __name__ == '__main__':
@app.route('/api/evaluate_models', methods=['GET'])
def get_model_evaluation():
    try:
        print("\n📊 Đang chạy bộ chấm điểm mô hình (Train 70% / Test 30%)...")
        print("⏳ Quá trình này sẽ chạy 100 Trees nên có thể mất 1-3 phút...")
        
        # Gọi hàm chấm điểm
        eval_results = evaluate_models(DEFAULT_LAT, DEFAULT_LON)
        
        return jsonify({
            "status": "success",
            "meta": {
                "latitude": DEFAULT_LAT,
                "longitude": DEFAULT_LON,
                "description": "Báo cáo chỉ số đánh giá mô hình học máy",
                "metrics_explained": {
                    "RMSE": "Càng nhỏ càng tốt (<2 là xuất sắc)",
                    "MAE": "Sai số tuyệt đối trung bình (Độ C, mm, km/h)",
                    "R2": "Độ khớp của mô hình (Càng gần 1 càng tốt)",
                    "MSE": "Sai số bình phương trung bình"
                }
            },
            "data": eval_results
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)