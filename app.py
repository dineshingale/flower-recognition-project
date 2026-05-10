from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Flower Recognition API")

MODEL = tf.keras.models.load_model("flower_model.keras")
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def preprocess_image(image_bytes):
    """
    Data Transformation: Scale and resize incoming images to match training.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    processed_image = preprocess_image(contents)

    predictions = MODEL.predict(processed_image)
    score = tf.nn.softmax(predictions[0])

    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = 100 * np.max(predictions)

    return {
        "flower_type": predicted_class,
        "confidence": f"{confidence:.2f}%"
    }

@app.get("/")
def home():
    return {"message": "Flower Recognition API is live!"}
