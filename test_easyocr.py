import easyocr
import cv2
import argparse
from PIL import Image, ImageTk
import tkinter as tk

# Charger le modèle easyocr
reader = easyocr.Reader(['en'])

# Fonction pour lire le texte et annoter l'image
def read_and_annotate(image_path):
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image {image_path}")
        return
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Utiliser easyocr pour lire le texte
    results = reader.readtext(gray)

    for (bbox, text, prob) in results:
        # Obtenir les coordonnées de la boîte englobante
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (int(top_left[0]), int(top_left[1]))
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
        
        # Dessiner la boîte englobante et le texte sur l'image
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convertir l'image de BGR à RGB pour l'affichage avec PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Créer une fenêtre tkinter pour afficher l'image
    root = tk.Tk()
    root.title("Text Detection")
    img_tk = ImageTk.PhotoImage(img_pil)
    label = tk.Label(root, image=img_tk)
    label.pack()
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Detection using easyocr")
    parser.add_argument('--imagepath', type=str, required=True, help='Path to the image file')
    
    args = parser.parse_args()
    read_and_annotate(args.imagepath)
