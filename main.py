import json 
import base64
import os
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Salak Smart Portal")
os.makedirs("dataset/flowers", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com"
)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    date_time: str = Form(...),
    farm_plot: str = Form(...),
    row_num: str = Form(...),
    tree_num: str = Form(...)
):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_location = f"dataset/flowers/{timestamp_str}_{file.filename}"
    
    image_bytes = await file.read()
    with open(file_location, "wb") as f:
        f.write(image_bytes)
        
    try:
        user_base64 = base64.b64encode(image_bytes).decode('utf-8')
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional botanical AI for Salak farming. Output valid JSON in Thai."},
                {"role": "user", "content": [
                    {"type": "text", "text": "วิเคราะห์ภาพถ่ายดอกสละนี้อย่างอิสระ: 1. จำแนกเพศ (ตัวผู้/ตัวเมีย) 2. ประเมินการผสมเกสร (ติด/ไม่ติด/ไม่แน่ใจ) พร้อม % ความมั่นใจในวงเล็บ 3. ให้เหตุผลประกอบ (pollination_reason) ว่าทำไมถึงติด ทำไมถึงไม่ติด หรือทำไมยังไม่พร้อม 4. สถานะโรค (ปกติ/เป็นโรค) 5. ระยะเวลารอคอย (waiting_duration): ต้องระบุให้ชัดเจนว่าเป็นเวลาอะไร เช่น 'พร้อมผสมในอีก 3 วัน' (ถ้ายังไม่พร้อม), 'ยังสามารถผสมได้อีก 2 วัน' (ถ้าบานพร้อมผสมแล้ว), หรือ '-' (ถ้าดอกหมดสภาพ/มีเหตุผลอื่นที่ทำให้ผสมไม่ได้แล้ว) 6. คำแนะนำสั้นๆ ส่งกลับเป็น JSON คีย์: gender, pollination_status, pollination_reason, disease_status, waiting_duration, recommendation"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{user_base64}"}}
                ]}
            ],
            temperature=0.6,
            response_format={"type": "json_object"}
        )
        ai_result = json.loads(response.choices[0].message.content)
    except Exception:
        ai_result = {}

    return {
        "date_time": date_time,
        "farm_plot": farm_plot,
        "row_num": row_num,
        "tree_num": tree_num,
        "gender": ai_result.get("gender", "ไม่ระบุ"),
        "pollination_status": ai_result.get("pollination_status", "ไม่แน่ใจ"),
        "pollination_reason": ai_result.get("pollination_reason", "-"),
        "disease_status": ai_result.get("disease_status", "ปกติ"),
        "waiting_duration": ai_result.get("waiting_duration", "-"),
        "recommendation": ai_result.get("recommendation", "-")
    }