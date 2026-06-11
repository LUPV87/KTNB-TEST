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
