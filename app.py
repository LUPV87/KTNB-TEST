import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ==========================================
# CẤU HÌNH TRANG ĐẦU TIÊN
# ==========================================
st.set_page_config(
    layout="wide",
    page_title="Hệ thống Phát Hiện Gian Lận Giao Dịch ngân hàng ",
    page_icon="🚩"
)

# ==========================================
# CÁC HÀM XỬ LÝ CACHE DỮ LIỆU
# ==========================================
@st.cache_data
def load_data(file_bytes, file_name):
    """Nạp dữ liệu từ file upload và kiểm tra định dạng"""
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_bytes)
        elif file_name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_bytes)
        else:
            return None, "Định dạng file không hỗ trợ. Vui lòng tải file .csv hoặc .xlsx"
        return df, None
    except Exception as e:
        return None, f"Lỗi khi đọc file: {str(e)}"

# ==========================================
# KHỞI TẠO SESSION STATE
# ==========================================
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None  # Notebook hiện tại chưa dùng scaler/encoder
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None

# ==========================================
# THÀNH PHẦN 1: SIDEBAR — VÙNG CẤU HÌNH
# ==========================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")
    
    # 1. Tải dữ liệu huấn luyện
    uploaded_file = st.file_uploader(
        "Tải lên tệp dữ liệu huấn luyện mẫu", 
        type=["csv", "xlsx"],
        help="Chọn tệp dữ liệu chứa các biến X_1 đến X_14 và biến mục tiêu 'default'"
    )
    
    st.divider()
    
    # 2. Lựa chọn mô hình AI
    model_option = st.selectbox(
        "Chọn thuật toán phân loại",
        options=["Random Forest", "Decision Tree", "Logistic Regression"],
        help="Chọn một trong ba thuật toán đã được thử nghiệm trong notebook gốc"
    )
    
    # 3. Tham số mô hình động theo thuật toán lựa chọn
    st.subheader("Tham số mô hình AI")
    
    model_params = {}
    if model_option == "Random Forest":
        model_params['n_estimators'] = st.slider(
            "Số lượng cây (n_estimators)", 
            min_value=10, max_value=200, value=100, step=10,
            help="Số lượng cây quyết định trong rừng."
        )
        model_params['random_state'] = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=32, step=1,
            help="Khóa ngẫu nhiên để tái hiện kết quả giống notebook."
        )
    elif model_option == "Decision Tree":
        model_params['random_state'] = st.number_input(
            "Trạng thái ngẫu nhiên (random_state)", 
            value=32, step=1,
            help="Khóa ngẫu nhiên cho cây quyết định."
        )
    elif model_option == "Logistic Regression":
        model_params['max_iter'] = st.slider(
            "Số vòng lặp tối đa (max_iter)", 
            min_value=100, max_value=1000, value=100, step=50,
            help="Tăng số vòng lặp nếu mô hình gặp cảnh báo không hội tụ (ConvergenceWarning)."
        )
        
    st.divider()
    
    # 4. Nút hành động huấn luyện duy nhất
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình", 
        type="primary", 
        use_container_width=True,
        help="Bấm để bắt đầu phân tách dữ liệu Train/Test và huấn luyện thuật toán."
    )

# ==========================================
# THÀNH PHẦN 2: HEADER — VÙNG ĐỊNH HƯỚNG
# ==========================================
st.title("🛡️ Ứng dụng Phát Hiện Giao Dịch Gian Lận & Rủi Rõ")
st.caption("Giải pháp phân loại nhị phân tự động nhận diện giao dịch bất thường (default = 1) dựa trên các thuộc tính ẩn danh X_1 đến X_14.")

if uploaded_file is None:
    st.info("💡 Vui lòng tải file dữ liệu (.csv hoặc .xlsx) ở thanh Sidebar bên trái để kích hoạt ứng dụng.")
    st.stop()
else:
    # Đọc dữ liệu thô qua hàm cache chung
    df, error_msg = load_data(uploaded_file, uploaded_file.name)
    if error_msg:
        st.error(error_msg)
        st.stop()
    st.caption(f"📁 Đang sử dụng tệp dữ liệu: `{uploaded_file.name}`")

st.divider()

# Xử lý Logic huấn luyện khi bấm nút ở Sidebar
if train_clicked:
    with st.spinner("Đang huấn luyện mô hình, vui lòng đợi trong giây lát..."):
        # Kiểm tra sự tồn tại của biến mục tiêu 'default'
        if 'default' not in df.columns:
            st.error("Lỗi Schema: Không tìm thấy cột biến mục tiêu 'default' trong dữ liệu!")
        else:
            # Phân tách X, y tương tự notebook gốc
            y = df['default']
            X = df.drop('default', axis=1)
            
            # Chia tập Train-Test với tỷ lệ 80-20, random_state=32 cố định theo notebook
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
            
            # Khởi tạo mô hình tương ứng theo cấu hình giao diện
            if model_option == "Random Forest":
                model = RandomForestClassifier(
                    n_estimators=model_params['n_estimators'], 
                    random_state=model_params['random_state']
                )
            elif model_option == "Decision Tree":
                model = DecisionTreeClassifier(random_state=model_params['random_state'])
            elif model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=model_params['max_iter'])
                
            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            
            # Dự báo trên tập kiểm định để lấy chỉ số đánh giá
            y_pred = model.predict(X_test)
            
            # Tính toán các chỉ tiêu kiểm định giống notebook công bố
            cm = confusion_matrix(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            
            # Lưu vào session_state để tái sử dụng xuyên suốt các Tab
            st.session_state.trained_model = model
            st.session_state.model_name = model_option
            st.session_state.evaluation_results = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'confusion_matrix': cm,
                'report': report_dict,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            st.toast(f"Đã huấn luyện thành công mô hình {model_option}!", icon="✅")

# ==========================================
# KHỞI TẠO HỆ THỐNG TAB NỘI DUNG CHÍNH
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Tổng quan dữ liệu", 
    "📈 Trực quan hóa biến", 
    "🎯 Kết quả kiểm định", 
    "🔮 Sử dụng mô hình dự báo"
])

# ------------------------------------------
# THÀNH PHẦN 3: TAB "TỔNG QUAN DỮ LIỆU"
# ------------------------------------------
with tab1:
    st.subheader("Phân tích cấu trúc dữ liệu thô")
    
    # Hiển thị Metrics kích thước tổng quan
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Số lượng dòng giao dịch", f"{df.shape[0]:,}")
    with col_m2:
        st.metric("Tổng số lượng cột", f"{df.shape[1]}")
    with col_m3:
        # Ước tính dung lượng file nạp vào RAM
        file_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        st.metric("Dung lượng tập dữ liệu (RAM)", f"{file_size_mb:.2f} MB")
        
    st.write("#### Xem trước 5 dòng dữ liệu đầu tiên (Head):")
    st.dataframe(df.head(), use_container_width=True)
    
    st.write("#### Thống kê mô tả các thuộc tính của mô hình (Describe):")
    # Lấy các biến đưa vào mô hình (X và y) theo yêu cầu đặc tả
    model_features = [c for c in df.columns if c.startswith('X_') or c == 'default']
    st.dataframe(df[model_features].describe(), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 4: TAB "TRỰC QUAN HÓA DỮ LIỆU"
# ------------------------------------------
with tab2:
    st.subheader("Phân tích phân phối trực quan các đặc trưng")
    
    # Chuẩn bị danh sách đặc trưng X
    x_features = [c for c in df.columns if c.startswith('X_')]
    
    # Sử dụng multiselect nếu danh sách biến quá nhiều (>4) để người dùng tự chọn phối hợp
    selected_features = st.multiselect(
        "Chọn tối đa các đặc trưng hình ảnh muốn hiển thị (Mặc định chọn sẵn 3 biến đầu):",
        options=x_features,
        default=x_features[:3] if len(x_features) >= 3 else x_features
    )
    
    # Khởi dựng lưới layout 2x2
    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    
    # 1. Biểu đồ Ưu tiên luôn vẽ: Biến mục tiêu y (default)
    with row1_col1:
        if 'default' in df.columns:
            class_counts = df['default'].value_counts().reset_index()
            class_counts.columns = ['Trạng thái', 'Số lượng']
            class_counts['Trạng thái'] = class_counts['Trạng thái'].map({0: 'Hợp lệ (0)', 1: 'Gian lận/Trễ hạn (1)'})
            
            fig_y = px.bar(
                class_counts, x='Trạng thái', y='Số lượng',
                color='Trạng thái', title="Phân phối nhãn Biến mục tiêu (default)",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_y, use_container_width=True)
        else:
            st.warning("Không tìm thấy biến mục tiêu 'default' để vẽ.")

    # 2. Vẽ động các biến X được chọn theo cấu trúc phân phối liên tục (Histogram)
    slots = [row1_col2, row2_col1, row2_col2]
    for idx, feature in enumerate(selected_features[:3]):
        with slots[idx]:
            fig_x = px.histogram(
                df, x=feature, color='default' if 'default' in df.columns else None,
                marginal="box", title=f"Phân phối tần suất của biến {feature}",
                barmode="overlay", color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_x, use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 5: TAB "KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH"
# ------------------------------------------
with tab3:
    st.subheader("Đánh giá độ chính xác thuật toán")
    
    # Điều phối luồng dữ liệu trống nếu chưa huấn luyện
    if st.session_state.evaluation_results is None:
        st.info("💡 Vui lòng thiết lập tham số thuật toán ở thanh Sidebar và bấm nút **'Huấn luyện mô hình'** để xem kết quả kiểm định chi tiết tại đây.")
    else:
        results = st.session_state.evaluation_results
        st.success(f"Đang hiển thị kết quả kiểm định của mô hình: **{st.session_state.model_name}**")
        
        # 1. Khối hiển thị các chỉ số cốt lõi (Metrics)
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("Độ chính xác (Accuracy)", f"{results['accuracy']:.2%}")
        with col_r2:
            st.metric("Độ chuẩn xác (Precision - Lớp 1)", f"{results['precision']:.2%}")
        with col_r3:
            st.metric("Độ bao phủ (Recall - Lớp 1)", f"{results['recall']:.2%}")
        with col_r4:
            st.metric("F1-Score (Lớp 1)", f"{results['f1']:.2%}")
            
        st.divider()
        
        # 2. Chia bố cục hai vùng biểu diễn Ma trận nhầm lẫn & Bảng phân loại
        col_viz1, col_viz2 = st.columns([1, 1])
        
        with col_viz1:
            st.write("##### Ma trận nhầm lẫn (Confusion Matrix):")
            cm_data = results['confusion_matrix']
            
            # Vẽ biểu đồ nhiệt Confusion Matrix bằng Plotly thay vì văn bản thô
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_data,
                x=['Dự báo Hợp lệ (0)', 'Dự báo Gian lận (1)'],
                y=['Thực tế Hợp lệ (0)', 'Thực tế Gian lận (1)'],
                colorscale='Blues',
                text=cm_data,
                texttemplate="%{text}",
                showscale=False
            ))
            fig_cm.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with col_viz2:
            st.write("##### Chi tiết báo cáo phân loại (Classification Report):")
            # Chuyển đổi định dạng báo cáo dạng dictionary sang DataFrame gọn gàng
            report_df = pd.DataFrame(results['report']).transpose()
            st.dataframe(report_df.style.format(precision=2), use_container_width=True)

# ------------------------------------------
# THÀNH PHẦN 6: TAB "SỬ DỤNG MÔ HÌNH DỰ BÁO"
# ------------------------------------------
with tab4:
    st.subheader("Chấm điểm và dự đoán rủi ro giao dịch mới")
    
    if st.session_state.trained_model is None:
        st.info("💡 Vui lòng hoàn tất bước huấn luyện mô hình ở thanh Sidebar để mở khóa tính năng chấm điểm rủi ro.")
    else:
        model = st.session_state.trained_model
        
        # Lựa chọn chế độ nhập đầu vào
        predict_mode = st.radio(
            "Chọn phương thức nhập dữ liệu đầu vào:",
            options=["Chế độ 1: Nhập trực tiếp thủ công", "Chế độ 2: Tải file dự báo hàng loạt (Batch Prediction)"],
            horizontal=True
        )
        
        # Trích xuất thông tin đặc trưng X để kiểm soát schema đầu vào
        feature_names = [f'X_{i}' for i in range(1, 15)] # Cố định từ X_1 đến X_14
        
        # --------------------------------------
        # CHẾ ĐỘ 1: NHẬP THỦ CÔNG QUA FORM
        # --------------------------------------
        if predict_mode == "Chế độ 1: Nhập trực tiếp thủ công":
            st.write("💬 Nhập các chỉ số giao dịch của một khách hàng cụ thể (Giá trị mặc định được lấy theo trung vị của tập dữ liệu mẫu):")
            
            with st.form("manual_prediction_form"):
                # Tạo lưới các ô nhập liệu 4 cột để tối ưu hóa không gian hiển thị
                form_cols = st.columns(4)
                input_data = {}
                
                for idx, feat in enumerate(feature_names):
                    col_slot = form_cols[idx % 4]
                    # Tính toán giá trị trung vị gốc làm mặc định giống đặc tả nghiệp vụ
                    default_val = float(df[feat].median()) if feat in df.columns else 0.0
                    min_val = float(df[feat].min()) if feat in df.columns else -100.0
                    max_val = float(df[feat].max()) if feat in df.columns else 100.0
                    
                    with col_slot:
                        input_data[feat] = st.number_input(
                            f"Chỉ số {feat}",
                            min_value=min_val, max_value=max_val,
                            value=default_val, format="%.4f"
                        )
                
                submit_predict = st.form_submit_button("🔍 Tiến hành phân tích rủi ro", type="primary")
                
            if submit_predict:
                # Chuyển đổi bản ghi nhập vào thành DataFrame đúng định dạng cột có tên đặc trưng
                input_df = pd.DataFrame([input_data])
                
                # Thực hiện dự đoán nhãn
                pred_class = model.predict(input_df)[0]
                
                # Hiển thị kết quả trực quan sinh động
                st.write("### Kết quả đánh giá từ AI:")
                if pred_class == 1:
                    st.error("🚨 **CẢNH BÁO:** Hệ thống phát hiện giao dịch này có **DẤU HIỆU GIAN LẬN / RỦI RO CAO** (Nhãn 1).")
                else:
                    st.success("✅ **AN TOÀN:** Giao dịch được thẩm định là **HỢP LỆ & KHÔNG CÓ RỦI RO** (Nhãn 0).")
                    
                # Hiện xác suất dự đoán (nếu mô hình hỗ trợ)
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    st.write(f"- Xác suất an toàn (Nhãn 0): `{probabilities[0]:.2%}`")
                    st.write(f"- Xác suất rủi ro (Nhãn 1): `{probabilities[1]:.2%}`")

        # --------------------------------------
        # CHẾ ĐỘ 2: TẢI FILE DỰ BÁO HÀNG LOẠT
        # --------------------------------------
        elif predict_mode == "Chế độ 2: Tải file dự báo hàng loạt (Batch Prediction)":
            st.write("📥 Tải lên tệp Excel hoặc CSV mới cần chấm điểm rủi ro hàng loạt. Tệp bắt buộc phải có đủ 14 cột từ `X_1` đến `X_14`.")
            
            new_file_uploader = st.file_uploader("Chọn tệp cần chấm điểm", type=["csv", "xlsx"], key="batch_uploader")
            
            if new_file_uploader is not None:
                new_df, err = load_data(new_file_uploader, new_file_uploader.name)
                if err:
                    st.error(err)
                else:
                    # Kiểm tra đối chứng schema cột
                    missing_cols = [col for col in feature_names if col not in new_df.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Sai cấu trúc tệp! Thiếu các cột dữ liệu bắt buộc sau: {missing_cols}")
                    else:
                        # Chỉ lọc lấy 14 đặc trưng cần thiết sắp xếp đúng thứ tự khi đưa vào model predict
                        predict_features_df = new_df[feature_names]
                        
                        # Dự đoán tập dữ liệu mới
                        batch_predictions = model.predict(predict_features_df)
                        
                        # Tích hợp kết quả dự báo ngược lại DataFrame xuất ra cho người dùng
                        output_df = new_df.copy()
                        output_df['Predicted_Default'] = batch_predictions
                        
                        if hasattr(model, "predict_proba"):
                            batch_probs = model.predict_proba(predict_features_df)
                            output_df['Probability_Risk_Class_1'] = batch_probs[:, 1]
                        
                        st.write("##### Xem trước kết quả xử lý hàng loạt:")
                        st.dataframe(output_df, use_container_width=True)
                        
                        # Tạo nút download kết quả chuẩn định dạng CSV utf-8-sig chống lỗi tiếng Việt font
                        csv_buffer = output_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 Tải xuống bảng kết quả kiểm định rủi ro (.CSV)",
                            data=csv_buffer,
                            file_name="ket_qua_du_bao_gian_lan.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
