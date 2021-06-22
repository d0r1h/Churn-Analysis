import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import pickle
origins = ["*"]
img_path='static/churn.jpg'
file_path='templates/index.html'
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pickle_in = open("churn_model.pkl","rb")
model=pickle.load(pickle_in)

@app.get("/")
async def root():
    return FileResponse(file_path)

@app.get("/image")
async def image():
    return FileResponse(img_path)

@app.get("/predict")
async def predict(gender_M:float, year:float, membership_category:float,complaint_status:float, feedback:float,age:float,days_since_last_login:float,avg_time_spent:float, avg_transaction_value:float,avg_frequency_login_days:float, points_in_wallet:float):
    data=np.array([gender_M, year, membership_category,complaint_status,feedback,age,days_since_last_login,avg_time_spent, avg_transaction_value,avg_frequency_login_days, points_in_wallet]).reshape(1, -1)
    prediction = model.predict(data)
    return {
        'p': int(prediction[0])
    }
    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)