from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np
import io

app = FastAPI()

model = YOLO('/var/www/html/best.pt')

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

@app.post("/detect/")
async def detect_cards(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = model.predict(source=image, save=True)

        result = results[0]

        img_processed = image.copy()
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            x1, y1, x2, y2 = box[:4]
            card_label = model.names[int(cls)]
            card_name = card_names_turkish.get(card_label, "Bilinmeyen Kart")
            cv2.rectangle(img_processed, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img_processed, f'{card_name} {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        output_path = "temp.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

        with open(output_path, mode="rb") as file:
            return StreamingResponse(io.BytesIO(file.read()), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5353)
