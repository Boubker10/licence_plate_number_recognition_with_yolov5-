import torch
import cv2
from pathlib import Path
import yaml
import argparse
import matplotlib.pyplot as plt
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
import pytesseract

model_path = 'yolov5m.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']


def convert_licence_to_text(boxes, img_cv2):
    global licences_text
    licences_text = []
    for (x1, y1, x2, y2) in boxes[:, :4]:  # Prendre seulement les quatre premières valeurs
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cropped_image = img_cv2[y1:y2, x1:x2]
        text_result = pytesseract.image_to_string(cropped_image, config='--psm 6')
        licences_text.append((text_result.strip(), (x1, y1, x2, y2)))  # Retourner aussi les coordonnées


def annotator(results, frame, class_names, Licence=False):
    xyxy = results.xyxy[0].cpu().numpy()
    confidences = results.xyxy[0][:, 4].cpu().numpy()
    class_ids = results.xyxy[0][:, 5].cpu().numpy().astype(int)

    if Licence:
        convert_licence_to_text(xyxy, frame)
        for text, (x1, y1, x2, y2) in licences_text:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        labels = [
            f"{class_names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(class_ids, confidences)
        ]

        for i, (x1, y1, x2, y2) in enumerate(xyxy[:, :4]):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = labels[i]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

def detect_vehicles(image_path,detect_license_plates=False):
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

    frame = annotator(results, img, model.names, Licence=detect_license_plates)


    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    parser.add_argument('--detect-license-plates', action='store_true', help='Enable license plate detection and OCR')
    
    args = parser.parse_args()
    detect_vehicles(args.imagepath, detect_license_plates=args.detect_license_plates)