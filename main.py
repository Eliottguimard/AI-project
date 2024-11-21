from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import os

object_detector = pipeline("object-detection", model="hustvl/yolos-tiny", framework="pt")

# Chemin du fichier image depuis la racine du projet
project_root = os.path.dirname(os.path.abspath(__file__))  # Racine du projet
image_path = os.path.join(project_root, "images", "motorcycle.jpg")  # Modifier selon l'organisation des dossiers

# Charger l'image
image = Image.open(image_path).convert("RGB")

results = object_detector(image)

print(results)

draw = ImageDraw.Draw(image)

box = results[0]['box']
xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
label = results[0]['label']

draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

font_path = "C:/Windows/Fonts/arial.ttf"
font = ImageFont.truetype(font_path, 40)

draw.text((xmin, ymin - 10), label, fill="red", font=font)

image.show()
