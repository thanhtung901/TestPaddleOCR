# from paddleocr import PaddleOCR

# # 默认使用 PP-OCRv5_medium
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="TestPaddleOCR\\PP-OCRv5_server_rec_infer",      
#     use_doc_orientation_classify=False,
#     engine="transformers",
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )

# result = ocr.predict("images_yolo_cropped\\15.jpg")

# for res in result:
#     res.print()

#     res.save_to_img("output")

from paddlex import create_model

model = create_model(
    # BẮT BUỘC: Tên đăng ký chính thức của PaddleX. 
    # (Nếu mô hình của bạn train từ base PP-OCRv4, hãy dùng tên v4 tương ứng)
    model_name="PP-OCRv5_mobile_rec", 
    
    # TRUYỀN ĐƯỜNG DẪN VÀO ĐÂY: Thư mục chứa model .pdiparams và .pdmodel của bạn
    model_dir="TestPaddleOCR\\PP-OCRv5_server_rec_infer" 
)
img_path = "TestPaddleOCR\\image7seg\\img_95.png"

# 3. Chạy dự đoán (Inference)
# Bạn có thể truyền vào một đường dẫn ảnh, một danh sách (list) các đường dẫn ảnh, 
# hoặc thậm chí là một mảng numpy (numpy.ndarray) nếu bạn đọc ảnh bằng cv2.
output = model.predict(img_path)

# 4. Xử lý và hiển thị kết quả
for res in output:
    # Cách 1: In trực tiếp kết quả ra màn hình (Console)
    res.print()