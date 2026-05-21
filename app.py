import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import sys

# Thêm thư mục hiện tại vào sys.path để import recommender
sys.path.append(str(Path.cwd()))
from recommender import load_artifact, recommend, get_time_slot, choose_top_item, summarize_customer_history

# Cấu hình trang
st.set_page_config(page_title="Food Category Recommender", page_icon="🍣", layout="wide")

# Tiêu đề ứng dụng
st.title("🍣 Food Category Recommender")
st.markdown("""
Ứng dụng này sử dụng thuật toán **RandomForest + Context Prior** để gợi ý loại món ăn phù hợp nhất cho khách hàng dựa trên ngữ cảnh đặt hàng.
""")

# Load model artifact
@st.cache_resource
def get_artifact():
    artifact_path = Path("models/category_recommender_v2/category_recommender_v2.joblib")
    if not artifact_path.exists():
        st.error(f"Không tìm thấy file model tại {artifact_path}. Vui lòng đảm bảo bạn đã chạy train model trước.")
        return None
    return load_artifact(artifact_path)

artifact = get_artifact()

if artifact:
    # Sidebar cho các tham số đầu vào
    st.sidebar.header("Tham số đầu vào")
    
    customer_id_input = st.sidebar.text_input("Customer ID (Để trống nếu là khách mới)", "")
    customer_id = float(customer_id_input) if customer_id_input.strip() != "" else None
    
    order_type = st.sidebar.selectbox("Loại đơn hàng", ["delivery", "collection"])
    
    day_of_week = st.sidebar.selectbox(
        "Ngày trong tuần", 
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: ["Thứ 2", "Thứ 3", "Thứ 4", "Thứ 5", "Thứ 6", "Thứ 7", "Chủ Nhật"][x]
    )
    
    hour = st.sidebar.slider("Giờ đặt hàng", 0, 23, 19)
    
    month = st.sidebar.slider("Tháng", 1, 12, int(artifact['default_month']))
    
    top_n = st.sidebar.number_input("Số lượng gợi ý (Top N)", 1, 20, 5)

    # Nút bấm dự đoán
    if st.sidebar.button("Gợi ý món ăn"):
        st.subheader("Kết quả gợi ý")
        
        # Hiển thị thông tin khách hàng
        history = summarize_customer_history(artifact, customer_id)
        customer_label = f"ID {int(customer_id)}" if history['customer_found'] else "Mới/Vãng lai"
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Khách hàng", customer_label)
        col2.metric("Tổng đơn", history['total_orders_all'])
        col3.metric("Độ phủ lịch sử", f"{history['coverage']:.1%}")
        
        # Thực hiện gợi ý
        with st.spinner('Đang tính toán...'):
            results = recommend(
                artifact=artifact,
                hour=hour,
                day_of_week=day_of_week,
                order_type=order_type,
                customer_id=customer_id,
                top_n=top_n,
                month=month,
            )
        
        # Hiển thị bảng kết quả
        st.table(results[['category', 'top_item', 'prob_pct']])
        
        # Biểu đồ xác suất
        st.bar_chart(results.set_index('category')['prob'])
else:
    st.info("Vui lòng đảm bảo các file `order.csv` và `order_item_final_best.csv` có sẵn để train model hoặc file model đã được tạo.")
