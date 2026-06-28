import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv

# โหลดค่า Environment
load_dotenv()

app = FastAPI(title="Salak Smart Portal")

# สร้างโฟลเดอร์สำหรับเก็บรูปถ้ายังไม่มี
os.makedirs("dataset/flowers", exist_ok=True)

# ให้ FastAPI รู้จักโฟลเดอร์ static
app.mount("/static", StaticFiles(directory="static"), name="static")

# ตั้งค่า OpenAI Client (ใช้ GitHub Models)
client = OpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com"
)

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...),
    date_time: str = Form(default="-"),
    farm_plot: str = Form(default="-"),
    row_num: str = Form(default="-"),    # 👈 แก้เป็น row_num ให้ตรงกับหน้าเว็บ
    tree_num: str = Form(default="-")    # 👈 แก้เป็น tree_num ให้ตรงกับหน้าเว็บ
):
    try:
        # 1. อ่านไฟล์รูปภาพและแปลงเป็น Base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # 2. คำสั่ง Prompt
        system_prompt = """
        คุณคือผู้เชี่ยวชาญด้านการเกษตรและเชี่ยวชาญเรื่อง "ดอกสละ" (Salak / Snake Fruit) โดยเฉพาะ
        หน้าที่ของคุณคือวิเคราะห์รูปภาพดอกสละที่เกษตรกรส่งมาให้ และตอบกลับเป็นรูปแบบ JSON เท่านั้น
        
        กฎการวิเคราะห์ที่ต้องจำให้แม่นยำ:
        1. เพศ (gender): ระบุว่าเป็น "ตัวผู้" หรือ "ตัวเมีย"
        2. ผลผสมเกสร (pollination_status):
           - กรณี "ดอกตัวผู้": วิเคราะห์ว่าเกสรพร้อมใช้งานไหม เช่น "เกสรพร้อมใช้งาน (95%)", "เกสรยังอ่อนเกินไป", หรือ "เกสรบานเต็มที่"
           - กรณี "ดอกตัวเมีย": วิเคราะห์ว่าพร้อมรับเกสรไหม เช่น "พร้อมผสมเกสร (90%)", "ผสมติดแล้ว", หรือ "เลยระยะเวลาผสม"
        3. เหตุผลประกอบ (pollination_reason): อธิบายจากลักษณะที่เห็นในภาพ
        4. สถานะโรค (disease_status): วิเคราะห์ว่า "ปกติ" หรือ "พบความผิดปกติ/เป็นโรค"
        5. ระยะเวลาเป้าหมาย (waiting_duration): เช่น "ใช้งานได้ทันที", "รออีก 2-3 วัน", หรือ "-"
        6. คำแนะนำ (recommendation): คำแนะนำสั้นๆ สำหรับชาวสวน

        รูปแบบ JSON ที่ต้องตอบกลับ (ห้ามพิมพ์ข้อความอื่นนอกจาก JSON):
        {
            "gender": "...",
            "pollination_status": "...",
            "pollination_reason": "...",
            "disease_status": "...",
            "waiting_duration": "...",
            "recommendation": "..."
        }
        """

        # 3. ส่งข้อมูลให้ AI ประมวลผล
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "ช่วยวิเคราะห์ดอกสละรูปนี้ให้หน่อยครับ"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            response_format={ "type": "json_object" },
            temperature=0.0, 
            seed=42          
        )

        # 4. แปลงข้อความ JSON ที่ AI ตอบกลับมาให้อยู่ในรูปแบบ Dictionary
        result_text = response.choices[0].message.content
        ai_result = json.loads(result_text)

        # 5. ส่งผลลัพธ์พิกัดกลับไปให้หน้าเว็บ (ใช้ชื่อตัวแปรเดียวกับหน้าเว็บเป๊ะๆ)
        ai_result["date_time"] = date_time
        ai_result["farm_plot"] = farm_plot
        ai_result["row_num"] = row_num
        ai_result["tree_num"] = tree_num
        
        return ai_result

    except Exception as e:
        print("Error:", e)
        return {
            "gender": "ไม่ทราบ",
            "pollination_status": "เกิดข้อผิดพลาด",
            "pollination_reason": f"ไม่สามารถวิเคราะห์ได้: {str(e)}",
            "disease_status": "-",
            "waiting_duration": "-",
            "recommendation": "กรุณาลองใหม่อีกครั้ง",
            "date_time": date_time,
            "farm_plot": farm_plot,
            "row_num": row_num,
            "tree_num": tree_num
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)