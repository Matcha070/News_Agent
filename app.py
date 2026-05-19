import os
import sys
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

# 🧠 ดึงตัวแปร workflow (LangGraph) มาจากไฟล์ agent.py เดิมของคุณ
from agent import workflow

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# 🔑 ดึงค่า Token ของ LINE มาจาก .env
LINE_SECRET = os.getenv("LINE_CHANNEL_SECRET")
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

if not LINE_SECRET or not LINE_TOKEN:
    print("❌ [Error] กรุณากรอก LINE_CHANNEL_SECRET และ LINE_CHANNEL_ACCESS_TOKEN ใน .env ก่อน")
    sys.exit(1)

line_bot_api = LineBotApi(LINE_TOKEN)
handler = WebhookHandler(LINE_SECRET)

# 🚀 Compile ตัว LangGraph ให้พร้อมใช้งาน
# (ตรวจสอบให้แน่ใจว่าใน agent.py มีบรรตรรัด app = workflow.compile() หรือยัง)
# หากใน agent.py ยังไม่มีตัวแปรที่ compile ให้เปลี่ยนบรรทัดล่างเป็น workflow.compile()
graph_agent = workflow.compile()


def process_and_reply(user_message: str, reply_token: str):
    """ฟังก์ชันทำงานเบื้องหลัง: ส่งข้อความเข้า LangGraph และยิงคำตอบกลับไปที่ LINE"""
    try:
        print(f"📥 [LINE Event] ได้รับข้อความ: '{user_message}'")
        
        # 🏃‍♂️ ส่งข้อความเข้าสู่ระบบห้ามล้อและลูปของ LangGraph ของเรา
        inputs = {"message": user_message}
        config = {"configurable": {"thread_id": "line_user_session"}} # ใส่ thread_id เผื่อใช้ระบบจำ
        
        result = graph_agent.invoke(inputs, config=config)
        
        # 📤 ดึงคำตอบสุดท้ายที่ AI สรุปและจัด Markdown สวยๆ ออกมา
        ai_reply = result.get("final_reply", "ขออภัยครับ ระบบไม่สามารถประมวลผลคำตอบได้")
        
        # 💬 ส่งข้อความกลับไปหาผู้ใช้ใน LINE
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=ai_reply)
        )
        print("📤 [LINE Reply] ส่งสรุปข่าวกลับไปหาผู้ใช้เรียบร้อยแล้ว!")
        
    except Exception as e:
        print(f"❌ [Error inside process_and_reply]: {e}")
        # หากระบบพัง กลัวผู้เจ้ารอนาน ให้ส่งข้อความแจ้งเตือนฉุกเฉิน
        try:
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="⚠️ เกิดข้อผิดพลาดในระบบจัดทำข่าวกรุณาลองใหม่อีกครั้งครับ")
            )
        except:
            pass

@app.post("/webhook")
async def line_webhook(request: Request, background_tasks: BackgroundTasks):
    """Endpoint หลักที่ LINE Webhook จะยิงเข้ามา"""
    signature = request.headers.get("X-Line-Signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Signature")

    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        # 🛡️ ส่งให้ Handler ตรวจสอบ Signature ว่ามาจาก LINE จริงๆ หรือไม่
        # โดยเราจะดักเอา Event ไว้ไปประมวลผลแบบ Background เพื่อป้องกัน LINE Timeout (ต้องตอบภายใน 1 วินาที)
        background_tasks.add_task(handler.handle, body_str, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid Signature")

    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    """ฟังก์ชันรับข้อความประเภท Text จากผู้ใช้"""
    user_message = event.message.text
    reply_token = event.reply_token
    
    # สั่งให้ทำงานเบื้องหลังทันทีเพื่อป้องกันบอทค้าง
    process_and_reply(user_message, reply_token)

if __name__ == "__main__":
    import uvicorn
    # รันเว็บเซิร์ฟเวอร์ที่ Port 8000
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)