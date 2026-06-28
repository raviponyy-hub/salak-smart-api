import os
import base64
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI
from dotenv import load_dotenv

# โหลดค่า Environment (เช่น GITHUB_TOKEN)
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
    row: str = Form(default="-"),    # 👈 เพิ่มช่องรับข้อมูล "แถวที่" แบบมีค่าเริ่มต้นกันพัง
    tree: str = Form(default="-")    # 👈 เพิ่มช่องรับข้อมูล "ต้นที่" แบบมีค่าเริ่มต้นกันพัง
):
    try:
        # 1. อ่านไฟล์รูปภาพและแปลงเป็น Base64
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')

        # 2. คำสั่ง Prompt ที่สอนให้ AI เป็นผู้เชี่ยวชาญสวนสละ
        system_prompt = """
        คุณคือผู้เชี่ยวชาญด้านการเกษตรและเชี่ยวชาญเรื่อง "ดอกสละ" (Salak / Snake Fruit) โดยเฉพาะ
        หน้าที่ของคุณคือวิเคราะห์รูปภาพดอกสละที่เกษตรกรส่งมาให้ และตอบกลับเป็นรูปแบบ JSON เท่านั้น
        
        กฎการวิเคราะห์ที่ต้องจำให้แม่นยำ (ห้ามตอบว่าดอกตัวผู้ไม่สามารถสร้างผลได้เด็ดขาด ให้เน้นที่ความพร้อมของเกสร):
        1. เพศ (gender): ระบุว่าเป็น "ตัวผู้" หรือ "ตัวเมีย"
        2. ผลผสมเกสร (pollination_status):
           - กรณี "ดอกตัวผู้": วิเคราะห์ว่าเกสรพร้อมใช้งานไหม เช่น "เกสรพร้อมใช้งาน (95%)", "เกสรยังอ่อนเกินไป", หรือ "เกสรบานเต็มที่"
           - กรณี "ดอกตัวเมีย": วิเคราะห์ว่าพร้อมรับเกสรไหม เช่น "พร้อมผสมเกสร (90%)", "ผสมติดแล้ว", หรือ "เลยระยะเวลาผสม"
        3. เหตุผลประกอบ (pollination_reason): อธิบายจากลักษณะที่เห็นในภาพให้เป็นประโยชน์กับชาวสวน เช่น 
           - ตัวผู้: "เกสรตัวผู้บานเต็มที่ สีแดง/น้ำตาลเข้ม มีละอองเกสรฟู พร้อมนำไปใช้งาน"
           - ตัวเมีย: "กาบดอกเปิดออกเต็มที่ สีแดงสด พร้อมรับละอองเกสรตัวผู้"
        4. สถานะโรค (disease_status): วิเคราะห์ว่า "ปกติ" หรือ "พบความผิดปกติ/เป็นโรค"
        5. ระยะเวลาเป้าหมาย (waiting_duration): เช่น "ใช้งานได้ทันที", "รออีก 2-3 วัน", หรือ "-"
        6. คำแนะนำ (recommendation): 
           - ตัวผู้: "สามารถตัดดอกตัวผู้นี้ไปเคาะละอองเกสรใส่ดอกตัวเมียได้เลย" หรือ "ควรรอให้เกสรฟูกว่านี้ก่อนตัด"
           - ตัวเมีย: "รีบนำเกสรตัวผู้มาเคาะผสมทันที" หรือ "รอกาบดอกเปิดอีกนิด"

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

        # 3. ส่งข้อมูลให้ AI ประมวลผล (ใช้ gpt-4o)
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

        # 5. ใส่ข้อมูลพิกัดส่งกลับไปให้หน้าเว็บ
        ai_result["date_time"] = date_time
        ai_result["farm_plot"] = farm_plot
        ai_result["row"] = row
        ai_result["tree"] = tree
        
        # 🛡️ โหมดป้องกันเหนียว: ดักซ้ำชื่อตัวแปรไว้ให้ทุกแบบที่หน้าเว็บของคุณอาจจะเรียกใช้
        # หน้าเว็บจะได้ไม่ขึ้น undefined และไม่ Crash ตอนแปลงข้อมูลเป็นไฟล์ Excel
        ai_result["flower_gender"] = ai_result.get("gender")
        ai_result["sex"] = ai_result.get("gender")
        
        ai_result["status"] = ai_result.get("pollination_status")
        ai_result["pollination_result"] = ai_result.get("pollination_status")
        ai_result["result"] = ai_result.get("pollination_status")
        
        ai_result["reason"] = ai_result.get("pollination_reason")
        
        ai_result["disease"] = ai_result.get("disease_status")
        ai_result["health_status"] = ai_result.get("disease_status")
        
        ai_result["waiting_time"] = ai_result.get("waiting_duration")
        ai_result["duration"] = ai_result.get("waiting_duration")
        ai_result["waiting"] = ai_result.get("waiting_duration")
        
        ai_result["advice"] = ai_result.get("recommendation")
        ai_result["suggestion"] = ai_result.get("recommendation")
        ai_result["suggest"] = ai_result.get("recommendation")
        
        return ai_result

    except Exception as e:
        print("Error:", e)
        error_result = {
            "gender": "ไม่ทราบ",
            "pollination_status": "เกิดข้อผิดพลาด",
            "pollination_reason": f"ไม่สามารถวิเคราะห์ได้: {str(e)}",
            "disease_status": "-",
            "waiting_duration": "-",
            "recommendation": "กรุณาลองใหม่อีกครั้ง",
            "date_time": date_time,
            "farm_plot": farm_plot,
            "row": row,
            "tree": tree
        }
        # ดักรหัสฝั่ง Error ด้วยเช่นกันป้องกัน Excel ค้างเวลาเกิด Error
        error_result["flower_gender"] = error_result["gender"]
        error_result["sex"] = error_result["gender"]
        error_result["status"] = error_result["pollination_status"]
        error_result["pollination_result"] = error_result["pollination_status"]
        error_result["result"] = error_result["pollination_status"]
        error_result["reason"] = error_result["pollination_reason"]
        error_result["disease"] = error_result["disease_status"]
        error_result["health_status"] = error_result["disease_status"]
        error_result["waiting_time"] = error_result["waiting_duration"]
        error_result["duration"] = error_result["waiting_duration"]
        error_result["waiting"] = error_result["waiting_duration"]
        error_result["advice"] = error_result["recommendation"]
        error_result["suggestion"] = error_result["recommendation"]
        error_result["suggest"] = error_result["recommendation"]
        return error_result

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)