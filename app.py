                pred_class = model.predict(input_df)[0]
                
                # Hiển thị kết quả trực quan sinh động
                st.write("### Kết quả đánh giá từ AI:")
                
                # Áp dụng in đậm và tô màu xanh dương (#0066cc) cho toàn bộ chuỗi văn bản thông báo kết quả theo hình ảnh mới tải lên
                if pred_class == 1:
                    st.write("<b style='color:#0066cc;'>🚨 CẢNH BÁO: Hệ thống phát hiện giao dịch này có DẤU HIỆU GIAN LẬN / RỦI RO CAO (Nhãn 1).</b>", unsafe_allow_html=True)
                else:
                    st.write("<b style='color:#0066cc;'>✅ AN TOÀN: Giao dịch được thẩm định là HỢP LỆ & KHÔNG CÓ RỦI RO (Nhãn 0).</b>", unsafe_allow_html=True)
                    
                # Hiện xác suất dự đoán (nếu mô hình hỗ trợ) kèm in đậm và tô màu xanh dương cho dữ liệu
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_df)[0]
                    # Sử dụng HTML trong st.write để in đậm và tô màu xanh dương (#0066cc) đúng theo ảnh đính kèm
                    st.write(f"- Xác suất an toàn (Nhãn 0): <b style='color:#0066cc;'>{probabilities[0]:.2%}</b>", unsafe_allow_html=True)
                    st.write(f"- Xác suất rủi ro (Nhãn 1): <b style='color:#0066cc;'>{probabilities[1]:.2%}</b>", unsafe_allow_html=True)

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
