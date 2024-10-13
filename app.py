from fastapi import FastAPI
import tensorflow as tf
import h5py
app = FastAPI()
model = tf.keras.models.load_model('model.h5')

@app.post("/predict/")
async def predict(data: dict):
    input_data = np.array(data['input'])
    prediction = model.predict(input_data)
    return {"prediction": prediction.tolist()}
