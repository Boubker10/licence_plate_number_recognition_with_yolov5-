import torch
import cv2
from pathlib import Path
import yaml
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk

model_path = 'yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']

def detect_vehicles(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return
    results = model(img)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    for i in range(len(labels)):
        if model.names[int(labels[i])] in vehicle_classes:
            x1, y1, x2, y2, conf = cords[i]
            x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{model.names[int(labels[i])]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    root = tk.Tk()
    root.title("Vehicle Detection")
    img_tk = ImageTk.PhotoImage(img_pil)
    label = tk.Label(root, image=img_tk)
    label.pack()
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle Detection using YOLOv5")
    parser.add_argument('--imagepath', type=str, required=True, help='Path to the image file')
    
    args = parser.parse_args()
    detect_vehicles(args.imagepath)