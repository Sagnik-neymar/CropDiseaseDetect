from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras

app = FastAPI(debug=True)

# Mounting a directory to serve static files (e.g., images)
app.mount("/static", StaticFiles(directory="images"), name="static")

# Reading the dl model
model = keras.models.load_model('./model/your_model.h5', compile=False)
model.compile(optimizer="Adam", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=[keras.metrics.CategoricalAccuracy(), 'accuracy'])

# Class labels for prediction
class_labels = [
    "Apple scab", "Apple Black rot", "Cedar apple rust", "Healthy Apple",
    "Healthy Blueberry", "Cherry  Powdery mildew", "Healthy Cherry", 
    "Corn Cercospora leaf spot Gray leaf spot", "Corn Common rust",
    "Corn Northern Leaf Blight", "Healthy Corn", "Grape Black Rot",
    "Grape Esca", "Grape Leaf blight", "Grape healthy", "Orange Haunglongbing",
    "Peach Bacterial spot", "Healthy Peach", "Bellpepper Bacterial spot",
    "Healthy Bellpepper", "Potato Early blight", "Potato Late blight",
    "Healthy Potato", "Healthy Raspberry", "Healthy Soyabean",
    "Squash Powdery mildew", "Strawberry Leaf scorch", "Healthy Strawberry",
    "Tomato Bacterial spot", "Tomato Early blight", "Tomato Late blight",
    "Tomato Leaf Mold", "Tomato Septoria leaf spot",
    "Tomato Spider mites/Two-spotted spider mite", "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus", "Tomato Tomato mosaic virus",
    "Healthy Tomato"
]

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type.startswith('image'):
        with open(f"images/{file.filename}", "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Image processing
        img_path = f"images/{file.filename}"
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, (224, 224))
        resized_img = np.expand_dims(resized_img, axis=0)
        resized_img = resized_img / 255.0  # Normalization
        yhat = model.predict(resized_img)
        predicted_class_index = int(np.argmax(yhat))
        predicted_class_label = class_labels[predicted_class_index]
        desc_img_path = "/static/desc_img/" + str(predicted_class_index) + ".png"

        with open("templates/index1.html", "r") as f:
            html_content = f.read()
            html_content = html_content.replace("embed_class_label", predicted_class_label)
            html_content = html_content.replace("embed_d_im_path", desc_img_path)
        return HTMLResponse(content=html_content, status_code=200)

    else:
        raise HTTPException(status_code=415, detail="Unsupported Media Type")

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
