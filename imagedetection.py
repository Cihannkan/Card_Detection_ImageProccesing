from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

model = YOLO('runs/detect/train/weights/best.pt')

card_names_turkish = {
    "Ah": "Kupa As",
    "Kh": "Kupa Papaz",
    "Qh": "Kupa Kız",
    "Jh": "Kupa Joker",
    "10h": "Kupa 10",
    "9h": "Kupa 9",
    "8h": "Kupa 8",
    "7h": "Kupa 7",
    "6h": "Kupa 6",
    "5h": "Kupa 5",
    "4h": "Kupa 4",
    "3h": "Kupa 3",
    "2h": "Kupa 2",
    "Ad": "Karo As",
    "Kd": "Karo Papaz",
    "Qd": "Karo Kız",
    "Jd": "Karo Joker",
    "10d": "Karo 10",
    "9d": "Karo 9",
    "8d": "Karo 8",
    "7d": "Karo 7",
    "6d": "Karo 6",
    "5d": "Karo 5",
    "4d": "Karo 4",
    "3d": "Karo 3",
    "2d": "Karo 2",
    "Ac": "Sinek As",
    "Kc": "Sinek Papaz",
    "Qc": "Sinek Kız",
    "Jc": "Sinek Joker",
    "10c": "Sinek 10",
    "9c": "Sinek 9",
    "8c": "Sinek 8",
    "7c": "Sinek 7",
    "6c": "Sinek 6",
    "5c": "Sinek 5",
    "4c": "Sinek 4",
    "3c": "Sinek 3",
    "2c": "Sinek 2",
    "As": "Maça As",
    "Ks": "Maça Papaz",
    "Qs": "Maça Kız",
    "Js": "Maça Joker",
    "10s": "Maça 10",
    "9s": "Maça 9",
    "8s": "Maça 8",
    "7s": "Maça 7",
    "6s": "Maça 6",
    "5s": "Maça 5",
    "4s": "Maça 4",
    "3s": "Maça 3",
    "2s": "Maça 2"
}

image_path = 'denem.jpg'

results = model.predict(source=image_path, save=True)

result = results[0]

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
    x1, y1, x2, y2 = box[:4]
    card_label = model.names[int(cls)]
    label = f'{card_names_turkish[card_label]} {conf:.2f}'
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2))
    ax.text(x1, y1, label, fontsize=12, color='white', bbox=dict(facecolor='red', alpha=0.5))

ax.imshow(img)
ax.axis('off')
plt.show()