from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add root directory to path to import TcmRecommender
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tcm_recommender import TcmRecommender
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Recommender
recommender = TcmRecommender()

# Serve images
image_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image")
app.mount("/images", StaticFiles(directory=image_path), name="images")

class VisualInput(BaseModel):
    x: int
    y: int
    view: str # 'front', 'back', 'side'

class SymptomInput(BaseModel):
    symptoms: List[str]
    click_pos: Optional[VisualInput] = None

@app.get("/")
async def root():
    return {"message": "TCM Acupoint Recommendation API is running"}

@app.get("/symptoms")
async def get_symptoms():
    """獲取症狀列表 API，供前端下拉選單呼叫"""
    return recommender.get_all_symptoms()

@app.post("/recommend")
async def recommend(input_data: SymptomInput):
    try:
        # TcmRecommender expects list of symptoms and dict for click_pos
        click_pos_dict = None
        if input_data.click_pos:
            click_pos_dict = {
                'x': input_data.click_pos.x,
                'y': input_data.click_pos.y,
                'view': input_data.click_pos.view
            }
        
        results = recommender.recommend(input_data.symptoms, click_pos_dict)
        return results
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
