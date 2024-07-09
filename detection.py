from ultralytics import YOLO
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

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model.predict(frame)

    result = results[0]

    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        x1, y1, x2, y2 = box[:4]
        card_label = model.names[int(cls)]
        label = f'{card_names_turkish.get(card_label, "Bilinmeyen Kart")} {conf:.2f}'
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Koordinatları tamsayıya dönüştür
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Card Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
