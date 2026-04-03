from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import re
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Khởi tạo ứng dụng Flask và cho phép CORS
app = Flask(__name__)
CORS(app)

# Cấu hình bộ nhớ tạm (Cache) và thử lại (Retry) cho API Open-Meteo
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

@app.route('/api/weather', methods=['GET'])
def get_weather_data():
    """
    Endpoint xử lý yêu cầu lấy dữ liệu thời tiết linh hoạt.
    Hỗ trợ nhận tham số qua URL (Query Parameters).
    """
    # 1. NHẬN VÀ KIỂM TRA CÁC THAM SỐ TỪ CLIENT (POSTMAN)
    api_type = request.args.get('api_type', 'forecast') 
    fields_param = request.args.get('fields')
    interval = request.args.get('interval', 'hourly') # Tần suất mặc định là 1 giờ
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    forecast_days = request.args.get('forecast_days')
    past_days = request.args.get('past_days')

    # Ràng buộc lỗi 1: Bắt buộc phải có danh sách các trường dữ liệu
    if not fields_param:
        return jsonify({
            "status": "error", 
            "message": "Thiếu tham số 'fields'. Vui lòng truyền các trường cần lấy, ví dụ: ?fields=temperature_2m,rain"
        }), 400

    requested_fields = fields_param.split(',')

    # 2. ĐỊNH TUYẾN URL TỰ ĐỘNG (ROUTING)
    # Danh sách các API Endpoint mà hệ thống hỗ trợ
    url_mapping = {
        'forecast': "https://api.open-meteo.com/v1/forecast",
        'archive': "https://archive-api.open-meteo.com/v1/archive",
        'air_quality': "https://air-quality-api.open-meteo.com/v1/air-quality"
    }

    url = url_mapping.get(api_type)
    
    # Ràng buộc lỗi 2: Sai loại API
    if not url:
        return jsonify({
            "status": "error", 
            "message": f"api_type '{api_type}' không tồn tại. Các lựa chọn hợp lệ: forecast, archive, air_quality."
        }), 400

    # 3. THIẾT LẬP THÔNG SỐ GỬI LÊN OPEN-METEO (PARAMS)
    params = {
        "latitude": 16.1667, 
        "longitude": 107.8333,
        "timezone": "Asia/Bangkok",
    }
    
    # Gán các trường dữ liệu vào đúng mức tần suất yêu cầu
    if interval in ['minutely_15', 'hourly', 'daily']:
        params[interval] = requested_fields
    else:
        # Ràng buộc lỗi 3: Sai tần suất
        return jsonify({
            "status": "error", 
            "message": f"interval '{interval}' không hợp lệ. Các lựa chọn: minutely_15, hourly, daily."
        }), 400

    # Cấu hình thời gian linh hoạt
    if start_date and end_date:
        params['start_date'] = start_date
        params['end_date'] = end_date
    if forecast_days:
        params['forecast_days'] = int(forecast_days)
    if past_days:
        params['past_days'] = int(past_days)

    try:
        # 4. GỌI API VÀ XỬ LÝ DỮ LIỆU
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Phân nhánh để lấy đúng cấu trúc thời gian tùy theo tần suất
        if interval == 'minutely_15':
            time_data = response.Minutely15()
        elif interval == 'hourly':
            time_data = response.Hourly()
        elif interval == 'daily':
            time_data = response.Daily()

        # Dựng cột ngày giờ (Date Axis)
        date_range = pd.date_range(
            start=pd.to_datetime(time_data.Time() + response.UtcOffsetSeconds(), unit="s", utc=True),
            end=pd.to_datetime(time_data.TimeEnd() + response.UtcOffsetSeconds(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=time_data.Interval()),
            inclusive="left"
        ).astype(str).tolist()

        # Khởi tạo từ điển dữ liệu với cột ngày giờ làm gốc
        data_dict = {"date": date_range}

        # Vòng lặp lấy dữ liệu động cho từng trường được yêu cầu
        for index, field in enumerate(requested_fields):
            data_dict[field] = time_data.Variables(index).ValuesAsNumpy().tolist()

        # 5. CHUYỂN ĐỔI THÀNH BẢNG (DATAFRAME) VÀ TRẢ VỀ JSON
        df = pd.DataFrame(data_dict)
        result_json = df.to_dict(orient='records') 

        return jsonify({
            "status": "success",
            "meta": {
                "api_used": api_type,
                "interval": interval,
                "url_fetched": url,
                "total_rows": len(result_json)
            },
            "data": result_json
        }), 200

    # ==== BẮT ĐẦU TỪ DÒNG NÀY (THAY THẾ KHỐI EXCEPT CŨ) ====
    except Exception as e:
        error_msg = str(e)
        
        # 1. Xử lý lỗi nhập sai tên trường (fields)
        if "invalid String value" in error_msg:
            # Dùng Regex để tự động bắt và trích xuất đúng cái từ bị gõ sai
            match = re.search(r"invalid String value ([a-zA-Z0-9_]+)", error_msg)
            
            if match:
                wrong_field = match.group(1) # Lấy ra cái tên bị sai (ví dụ: rain1, abc...)
                return jsonify({
                    "status": "error", 
                    "message": f"Không có dữ liệu cho trường: '{wrong_field}'."
                }), 400
            else:
                # Trường hợp không bắt được từ cụ thể
                return jsonify({
                    "status": "error", 
                    "message": "Trường dữ liệu yêu cầu không tồn tại."
                }), 400
                
        # 2. Xử lý lỗi sai tần suất (ví dụ: dùng biến của 15 phút cho interval 1 ngày)
        elif "Data corrupted" in error_msg:
             return jsonify({
                "status": "error", 
                # Tự động lấy biến interval người dùng vừa nhập để báo lỗi
                "message": f"Các trường dữ liệu được yêu cầu không tương thích với tần suất interval='{interval}'."
            }), 400
            
        # 3. Xử lý các lỗi hệ thống không lường trước khác
        return jsonify({
            "status": "error", 
            "message": f"Lỗi hệ thống: {error_msg}"
        }), 500

if __name__ == '__main__':
   
    app.run(debug=True, port=5000)