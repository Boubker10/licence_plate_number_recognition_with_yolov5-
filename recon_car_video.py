import torch
import cv2
from pathlib import Path
import argparse
from PIL import Image, ImageTk
import tkinter as tk

model_path = 'yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

def detect_vehicles_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        
        for i in range(len(labels)):
            if model.names[int(labels[i])] in vehicle_classes:
                x1, y1, x2, y2, conf = cords[i]
                x1, y1, x2, y2 = int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{model.names[int(labels[i])]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Vehicle Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection using YOLOv5")
    parser.add_argument('--videopath', type=str, required=True, help='Path to the video file')
    
    args = parser.parse_args()
    detect_vehicles_video(args.videopath)
