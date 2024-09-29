import torch
import cv2
from deep_sort_pytorch.deep_sort import DeepSort
import json

# Load mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Khởi tạo Deep SORT
deepsort = DeepSort('ckpt.t7')

object_tracking = {}

def detect_and_track(video_path, skip_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0  # Biến để đếm số lượng frame đã đọc

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Chỉ xử lý mỗi skip_frames frame
        if frame_count % skip_frames == 0:
            results = model(frame)

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
                classes.append(int(cls.item()))  

            outputs, mask_outputs = deepsort.update(torch.Tensor(bbox_xywh), torch.Tensor(confs), classes, frame)

            # Vẽ bounding box, nhãn lớp và ID của đối tượng
            if outputs is not None and len(outputs) > 0:
                for output in outputs:
                    x1, y1, x2, y2, track_cls, track_id = output
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID {track_id}, Class {track_cls}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    area = (x2-x1) * (y2-y1)
                    object_id = str(track_id)
                    position = [str(x1),str(y1),str(x2),str(y2)]
                    if object_id not in object_tracking:
                        object_tracking.update({object_id: {'area':[str(area)], 'position': [position]}})
                    else:
                        object_tracking[object_id]['area'].append(str(area))
                        object_tracking[object_id]['position'].append(position)


            print(object_tracking)
            cv2.imshow('Multiple Object Tracking', frame)

        # Tăng biến đếm frame
        frame_count += 1

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Chạy chương trình với video, bỏ qua mỗi 5 frame
detect_and_track('highway.mp4', skip_frames=5)
# Lưu kết quả theo dõi đối tượng vào file JSON
with open('result.json', 'w') as fw:
    json.dump(object_tracking, fw, indent=4)