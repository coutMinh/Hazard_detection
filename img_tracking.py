import torch
import cv2
import os
from deep_sort_pytorch.deep_sort import DeepSort

# Load mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Khởi tạo Deep SORT
deepsort = DeepSort('ckpt.t7')
def detect_and_track_from_folder(folder_path):
    # Lấy danh sách các tệp ảnh trong thư mục và sắp xếp theo thứ tự
    image_files = sorted([os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith(('.png', '.jpg', '.jpeg'))])

    for image_path in image_files:
        # Đọc từng frame từ thư mục ảnh
        frame = cv2.imread(image_path)
        if frame is None:
            continue

        # Phát hiện đối tượng bằng YOLO
        results = model(frame)

        # Lấy các bounding box, confidence và nhãn (classes)
        bbox_xywh = []
        confs = []
        classes = []
        for det in results.xyxy[0]:  # x_min, y_min, x_max, y_max, conf, cls
            x_min, y_min, x_max, y_max, conf, cls = det
            bbox_xywh.append([x_min + (x_max - x_min) / 2,  # x_center
                              y_min + (y_max - y_min) / 2,  # y_center
                              x_max - x_min,  # width
                              y_max - y_min])  # height
            confs.append(conf.item())
            classes.append(int(cls.item()))  # Lưu lại nhãn lớp của đối tượng

        # Theo dõi đối tượng bằng Deep SORT
        outputs, mask_outputs = deepsort.update(torch.Tensor(bbox_xywh), torch.Tensor(confs), classes, frame)

        # Vẽ bounding box, nhãn lớp và ID của đối tượng
        if outputs is not None and len(outputs) > 0:
            for output in outputs:
                x1, y1, x2, y2, track_cls, track_id = output
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}, Class {track_cls}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Hiển thị kết quả cho từng frame
        cv2.imshow('Multiple Object Tracking', frame)
        save_path = 'result/' + image_path.split('\\')[-1]
        cv2.imwrite(save_path, frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

# Chạy chương trình với folder chứa các frame
detect_and_track_from_folder('img')
