from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import datetime
import os
import json
import math
import re



app = Flask(__name__)
CORS(app)

# TẮT TÍNH NĂNG TỰ ĐỘNG SẮP XẾP A-Z CỦA FLASK
app.json.sort_keys = False



# 1. TỪ ĐIỂN TỌA ĐỘ 63 TỈNH THÀNH & HÀM XỬ LÝ VỊ TRÍ

VIETNAM_PROVINCES = {
    "An Giang": {"lat": 10.5216, "lon": 105.1259}, "Bà Rịa - Vũng Tàu": {"lat": 10.4973, "lon": 107.1683},
    "Bạc Liêu": {"lat": 9.2941, "lon": 105.7278}, "Bắc Giang": {"lat": 21.2731, "lon": 106.1946},
    "Bắc Kạn": {"lat": 22.1470, "lon": 105.8348}, "Bắc Ninh": {"lat": 21.1861, "lon": 106.0763},
    "Bến Tre": {"lat": 10.2385, "lon": 106.3774}, "Bình Dương": {"lat": 11.1661, "lon": 106.6263},
    "Bình Định": {"lat": 13.7820, "lon": 109.2197}, "Bình Phước": {"lat": 11.7517, "lon": 106.9230},
    "Bình Thuận": {"lat": 10.9333, "lon": 108.1000}, "Cà Mau": {"lat": 8.9114, "lon": 105.1500},
    "Cao Bằng": {"lat": 22.6667, "lon": 106.2500}, "Cần Thơ": {"lat": 10.0222, "lon": 105.7145},
    "Đà Nẵng": {"lat": 16.0544, "lon": 108.2022}, "Đắk Lắk": {"lat": 12.6667, "lon": 108.0333},
    "Đắk Nông": {"lat": 12.0000, "lon": 107.6917}, "Điện Biên": {"lat": 21.3833, "lon": 103.0167},
    "Đồng Nai": {"lat": 10.9412, "lon": 106.8202}, "Đồng Tháp": {"lat": 10.4608, "lon": 105.6358},
    "Gia Lai": {"lat": 13.9833, "lon": 108.0000}, "Hà Giang": {"lat": 22.8167, "lon": 104.9833},
    "Hà Nam": {"lat": 20.5312, "lon": 105.9234}, "Hà Nội": {"lat": 21.0285, "lon": 105.8542},
    "Hà Tĩnh": {"lat": 18.3333, "lon": 105.9000}, "Hải Dương": {"lat": 20.9388, "lon": 106.3214},
    "Hải Phòng": {"lat": 20.8449, "lon": 106.6881}, "Hậu Giang": {"lat": 9.7828, "lon": 105.4711},
    "Hòa Bình": {"lat": 20.8167, "lon": 105.3333}, "Hưng Yên": {"lat": 20.6483, "lon": 106.0506},
    "Khánh Hòa": {"lat": 12.2388, "lon": 109.1967}, "Kiên Giang": {"lat": 10.0125, "lon": 105.0809},
    "Kon Tum": {"lat": 14.3500, "lon": 108.0000}, "Lai Châu": {"lat": 22.3833, "lon": 103.4500},
    "Lạng Sơn": {"lat": 21.8478, "lon": 106.7581}, "Lào Cai": {"lat": 22.4833, "lon": 103.9667},
    "Lâm Đồng": {"lat": 11.9465, "lon": 108.4419}, "Long An": {"lat": 10.5333, "lon": 106.4000},
    "Nam Định": {"lat": 20.4333, "lon": 106.1667}, "Nghệ An": {"lat": 18.6667, "lon": 105.6667},
    "Ninh Bình": {"lat": 20.2539, "lon": 105.9750}, "Ninh Thuận": {"lat": 11.5667, "lon": 108.9833},
    "Phú Thọ": {"lat": 21.3333, "lon": 105.2167}, "Phú Yên": {"lat": 13.0833, "lon": 109.3000},
    "Quảng Bình": {"lat": 17.4833, "lon": 106.6000}, "Quảng Nam": {"lat": 15.5833, "lon": 108.0000},
    "Quảng Ngãi": {"lat": 15.1167, "lon": 108.8000}, "Quảng Ninh": {"lat": 20.9500, "lon": 107.0833},
    "Quảng Trị": {"lat": 16.7500, "lon": 107.1833}, "Sóc Trăng": {"lat": 9.6000, "lon": 105.9667},
    "Sơn La": {"lat": 21.3333, "lon": 103.9000}, "Tây Ninh": {"lat": 11.3000, "lon": 106.1000},
    "Thái Bình": {"lat": 20.4500, "lon": 106.3333}, "Thái Nguyên": {"lat": 21.5942, "lon": 105.8482},
    "Thanh Hóa": {"lat": 19.8000, "lon": 105.7667}, "Thừa Thiên Huế": {"lat": 16.4667, "lon": 107.5833},
    "Tiền Giang": {"lat": 10.3541, "lon": 106.3571}, "Hồ Chí Minh": {"lat": 10.7626, "lon": 106.6602},
    "Trà Vinh": {"lat": 9.9333, "lon": 106.3500}, "Tuyên Quang": {"lat": 21.8167, "lon": 105.2167},
    "Vĩnh Long": {"lat": 10.2500, "lon": 105.9667}, "Vĩnh Phúc": {"lat": 21.3000, "lon": 105.6000},
    "Yên Bái": {"lat": 21.7167, "lon": 104.8833}
}

def get_closest_province(lat, lon):
    """Tính khoảng cách Haversine để map tọa độ client sang Tỉnh gần nhất"""
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    closest = min(VIETNAM_PROVINCES.keys(), key=lambda p: haversine(lat, lon, VIETNAM_PROVINCES[p]["lat"], VIETNAM_PROVINCES[p]["lon"]))
    return closest


# 2. TẢI DỮ LIỆU DỰ ĐOÁN AI VÀ BÁO CÁO ĐỘ TIN CẬY VÀO RAM

csv_path = os.path.join(os.getcwd(), "predict", "Ultimate", "DuDoan_Ultimate_1.csv") 
eval_path = os.path.join(os.getcwd(), "reports", "Ultimate", "BaoCao_Ultimate_1.json")

try:
    df_predictions = pd.read_csv(csv_path)
    df_predictions['datetime'] = df_predictions['datetime'].astype(str)
    print(f"✅ Đã tải dữ liệu dự báo từ: {csv_path}")
except Exception as e:
    print(f"⚠️ Không tìm thấy file CSV. Lỗi: {e}")
    df_predictions = pd.DataFrame()

try:
    with open(eval_path, "r", encoding="utf-8") as f:
        model_metrics = json.load(f)
    print("✅ Đã tải báo cáo độ tin cậy AI.")
except Exception as e:
    print(f"⚠️ Không tìm thấy file JSON: {e}")
    model_metrics = {}

DEFAULT_LAT = 10.7626
DEFAULT_LON = 106.6602


# 3. THIẾT LẬP KẾT NỐI OPEN-METEO API

cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

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


# ROUTE 1: LẤY DỮ LIỆU THỜI TIẾT TỪ OPEN-METEO (THỜI GIAN THỰC/QUÁ KHỨ)

@app.route('/api/weather', methods=['GET'])
def get_weather_data():
    try:
        api_type = request.args.get('api_type', 'forecast') 
        interval = request.args.get('interval', 'hourly')
        fields_param = request.args.get('fields') 
        lat = request.args.get('latitude', default=DEFAULT_LAT, type=float)
        lon = request.args.get('longitude', default=DEFAULT_LON, type=float)
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        forecast_days = request.args.get('forecast_days')
        past_days = request.args.get('past_days')

        if interval not in ALL_FIELDS:
            return jsonify({"status": "error", "message": "Interval không hợp lệ"}), 400

        all_fields_for_interval = ALL_FIELDS[interval]
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
            interval: all_fields_for_interval
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
            "data": result_json,
            "meta": {
                "latitude": lat, 
                "longitude": lon, 
                "api_used": api_type, 
                "interval": interval,
                "client_requested_fields": client_requested_fields
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi hệ thống: {str(e)}"}), 500


# =====================================================================
# ROUTE 2: LẤY DỮ LIỆU DỰ ĐOÁN TỪ MACHINE LEARNING (CẬP NHẬT TÁCH INPUT)
# =====================================================================
@app.route('/api/predict_core', methods=['GET'])
def get_core_predictions():
    try:
        lat = request.args.get('latitude', default=DEFAULT_LAT, type=float)
        lon = request.args.get('longitude', default=DEFAULT_LON, type=float)
        target_date_raw = request.args.get('date')
        target_hour_raw = request.args.get('hour')
        client_requested_fields = request.args.get('fields', 'all') 
        province_name = get_closest_province(lat, lon)
        if not target_date_raw or target_hour_raw is None:
            return jsonify({
                "status": "error", 
                "message": "Thiếu tham số. Vui lòng cung cấp đủ 'date' (VD: 2026-04-22 hoặc 22-04-2026) và 'hour' (VD: 12)."
            }), 400

        
       
        try:
            # BƯỚC XỬ LÝ 1: Nắn ngày chuẩn
            parsed_date = pd.to_datetime(target_date_raw, dayfirst=True)
            formatted_date = parsed_date.strftime('%Y-%m-%d')
            
            # BƯỚC XỬ LÝ 2: Nắn giờ chuẩn
            hour_match = re.search(r'\d+', str(target_hour_raw))
            if not hour_match:
                raise ValueError(f"Không tìm thấy số trong tham số hour: '{target_hour_raw}'")
            
            hour_int = int(hour_match.group())
            if not (0 <= hour_int <= 23):
                raise ValueError(f"Giờ ({hour_int}) vi phạm giới hạn 0-23")
            formatted_hour = f"{hour_int:02d}:00:00"
            
            target_datetime = f"{formatted_date} {formatted_hour}"
            
        except Exception as e:
            # MỞ KHÓA BẮT LỖI: Nhả chính xác lý do hỏng ra Postman để bạn dễ sửa
            return jsonify({
                "status": "error", 
                "message": "Định dạng ngày/giờ không hợp lệ.",
                "chi_tiet_loi_he_thong": str(e),
                "du_lieu_ban_da_gui": {
                    "date": target_date_raw, 
                    "hour": target_hour_raw
                }
            }), 400

        if df_predictions.empty:
            return jsonify({"status": "error", "message": "Dữ liệu AI chưa sẵn sàng."}), 500

        row_data = df_predictions[(df_predictions['province'] == province_name) & (df_predictions['datetime'] == target_datetime)]

        if row_data.empty:
            return jsonify({"status": "error", "message": f"Không có dữ liệu cho {province_name} lúc {target_datetime}."}), 404

        row = row_data.iloc[0]

        # Đóng gói Response trả về Frontend 
        response_data = {
            "status": "success",
            "data": {
                "datetime": target_datetime,
                "province": province_name,
                "temperature_2m": {
                    "Random_Forest": row.get("temperature_2m_Random_Forest"),
                    "Linear_Regression": row.get("temperature_2m_Linear_Regression"),
                    "GBT": row.get("temperature_2m_GBT")
                },
                "relative_humidity_2m": {
                    "Random_Forest": row.get("relative_humidity_2m_Random_Forest"),
                    "Linear_Regression": row.get("relative_humidity_2m_Linear_Regression"),
                    "GBT": row.get("relative_humidity_2m_GBT")
                },
                "wind_speed_10m": {
                    "Random_Forest": row.get("wind_speed_10m_Random_Forest"),
                    "Linear_Regression": row.get("wind_speed_10m_Linear_Regression"),
                    "GBT": row.get("wind_speed_10m_GBT")
                },
                "precipitation": {
                    "Random_Forest": row.get("precipitation_Random_Forest"),
                    "Linear_Regression": row.get("precipitation_Linear_Regression"),
                    "GBT": row.get("precipitation_GBT")
                }
            },
            "meta": {
                "client_request": {
                    "latitude": lat, 
                    "longitude": lon,
                    "date_input": target_date_raw, 
                    "hour_input": target_hour_raw,
                    "fields_requested": client_requested_fields,
                    "province_matched": province_name,
                },
                # "model_accuracy": model_metrics     đây là phần 4 độ tin cậy cho từng report
            }
        }
        return jsonify(response_data), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Lỗi hệ thống AI: {str(e)}"}), 500        

if __name__ == '__main__':
    app.run(debug=True, port=5000)