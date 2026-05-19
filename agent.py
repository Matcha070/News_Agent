import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import END, START, StateGraph
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from openai import OpenAI

load_dotenv()

# ==========================================
# 1. โครงสร้าง State (รับแค่ Input Message ไม่เก็บประวัติข้ามรอบ)
# ==========================================
class State(TypedDict):
    message: str                   # 📥 รับ Input เป็นข้อความ String จากผู้ใช้รอบต่อรอบ
    messages: List[Dict[str, Any]] # 🧠 ใช้ประมวลผลชั่วคราวภายใน Loop (โดนล้างไพ่ทุกครั้งที่ขึ้นเทิร์นใหม่)
    final_reply: str               # 📤 คำตอบสุดท้ายส่งกลับไปแสดงผลบนหน้าจอ

# ==========================================
# 2. เครื่องมือดึงข่าว (ย้ายกลับมาใช้ NewsData.io)
# ==========================================
def get_news_from_url(url):
    if not url: return "ไม่มี URL ข่าว"
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text.strip()
        if not text:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = " ".join([p.get_text().strip() for p in paragraphs])
        if not text: return "ไม่สามารถดึงเนื้อหาข่าวได้"
        return text[:500] + "..." 
    except Exception as e:
        return f"(เกิดข้อผิดพลาดในการดึงเนื้อหา: {e})"

def fetch_news_tool(query=None, category=None, language="en", size=1, sort_by="relevancy", datatype="news"):
    """ฟังก์ชันดึงข่าวจาก NewsData.io"""
    print(f"🛠️ [Tool Executed] AI กำลังดึงข่าว | Query: {query}, Category: {category}, Lang: {language}")
    
    api_key = os.getenv("NEWS_API_KEY") # 🔑 ใช้คีย์ตัวเดิมของ NewsData.io
    url = "https://newsdata.io/api/1/latest"
    
    params = {
        "apikey": api_key, 
        "q": query, 
        "category": category, 
        "language": language, 
        "size": size,
        "sort": sort_by,
        "datatype": datatype
    }
    
    if query and str(query).strip() not in ["", "None", "null"]:
        params["q"] = query.strip()
        
    # 3. ตรวจสอบ Category: จะใส่คีย์ "category" เฉพาะตอนที่มีการส่งค่ามาจริงๆ เท่านั้น
    if category and str(category).strip() not in ["", "None", "null"]:
        params["category"] = category.strip()

    try:
        response = requests.get(url, params=params)
        response.raise_for_status() 
        news_data = response.json()
        
        if news_data.get("status") == "success":
            articles = news_data.get("results", [])
            if not articles:
                return "ไม่พบข่าวตามเงื่อนไขที่ค้นหาในช่วง 48 ชั่วโมงที่ผ่านมา"
            
            result_text = f"ข้อมูลข่าวดิบ {len(articles)} ข่าว:\n\n"
            for index, article in enumerate(articles, start=1):
                title = article.get("title", "ไม่มีหัวข้อข่าว")
                content = get_news_from_url(article.get("link", ""))
                news_url = article.get("link", "")
                result_text += f"{index}. หัวข้อ: {title}\nเนื้อหา: {content}\nที่มา: {news_url}\n\n"
            
            return result_text
        return "พบข้อผิดพลาดจาก API ข่าว"
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการดึงข้อมูล: {e}"

# ==========================================
# 3. กำหนดคู่มือ (Schema) ของ NewsData.io ให้ AI
# ==========================================
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "fetch_news",
            "description": "ใช้สำหรับดึงข้อมูลข่าวสารล่าสุดและเหตุการณ์ปัจจุบันจาก NewsData.io API รองรับทั้งข่าวไทยและต่างประเทศ (ข้อมูลย้อนหลัง 48 ชั่วโมง)",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "คำค้นหาสำหรับ NewsData.io API\n\n"
                            "## กฎการสร้าง Query (สำคัญมาก)\n"
                            "1. **ใช้ภาษาไทยเสมอ**: ส่ง query เป็นภาษาไทยโดยตรง ห้ามแปลเป็นภาษาอังกฤษ\n"
                            "   - ✅ 'น้ำท่วม'  ❌ 'flood'\n"
                            "   - ✅ 'รถไฟ'  ❌ 'train'\n"
                            "   - ✅ 'น้ำมัน'  ❌ 'oil'\n"
                            "   - ✅ 'มักกะสัน'  ❌ 'Makkasan'\n"
                            "2. **ใช้ 1-2 คำสำคัญเท่านั้น (บังคับเด็ดขาด)**: สกัดเฉพาะ noun หรือชื่อเฉพาะที่สำคัญที่สุด ห้ามเกิน 2 คำ ห้ามใส่บริบทหรือคำขยายเพิ่ม\n"
                            "   - ✅ 'น้ำท่วม'  ❌ 'น้ำท่วมกรุงเทพในฤดูฝน'\n"
                            "   - ✅ 'รถไฟชน'  ❌ 'รถไฟชนกันที่มักกะสันเสียหาย'\n"
                            "   - ✅ 'มักกะสัน'  ❌ 'มักกะสันโครงการพัฒนาพื้นที่'\n"
                            "3. **ถ้าไม่เจอในรอบแรก**: ลองคำที่กว้างกว่า หรือใช้ OR เชื่อมคำพ้องความหมาย เช่น 'รถไฟ OR รถด่วน' (นับเป็น 1 query)\n"
                            "4. **ถ้าไม่มีคำค้นเฉพาะ**: ห้ามส่ง parameter นี้ (ละไว้เลย)\n"
                            "⚠️ สรุป: query ต้องเป็นภาษาไทย และมีคำไม่เกิน 2 คำเสมอ\n"
                        )
                    },
                    "category": {
                        "type": "string",
                        "enum": ["business", "entertainment", "environment", "food", "health", "politics", "science", "sports", "technology", "world", "top"],
                        "description": (
                            "หมวดหมู่ของข่าวสารที่ต้องการคัดกรอง ต้องเลือกจากค่าใน Enum นี้เท่านั้น "
                            "💡 คำแนะนำ: หากผู้ใช้ถามหาข่าวเด่น ข่าววันนี้ หรือข่าวล่าสุดที่ไม่ได้ระบุหมวดหมู่ "
                            "ให้เลือกใช้ค่า 'top' เสมอ"
                        )
                    },
                    "language": {
                        "type": "string",
                        "enum": ["th", "en"],
                        "description": "ภาษาของข่าว บังคับใช้ 'th' สำหรับข่าวในประเทศไทย และใช้ 'en' สำหรับข่าวต่างประเทศหรือข่าวโลก"
                    },
                    "size": {
                        "type": "integer",
                        "description": (
                            "จำนวนข่าวที่ต้องการดึง (ค่าเริ่มต้น 1, สูงสุด 10)"
                            "หากผู้ใช้ถามหาข่าวทั่วไปหรือข่าวล่าสุดโดยไม่มีคำค้นเฉพาะเจาะจง ให้ตั้งค่าเป็น 3-5 เพื่อให้ได้ภาพรวมข่าวที่หลากหลายมากขึ้น"
                            "หากผู้ใช้ถามหาข่าวเฉพาะเจาะจง เช่น 'มีข่าวเกี่ยวกับน้ำท่วมในกรุงเทพไหม?' ให้ตั้งค่าเป็น 1-2 เพื่อเน้นความเฉพาะเจาะจงมากขึ้น"
                        )
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevancy", "source", "fetched_at"],
                        "description": (
                            "จัดเรียงลำดับข่าวที่ได้มา โดยมีตัวเลือกดังนี้: 'relevancy', 'source', 'fetched_at' (วันที่ดึงข้อมูล) "
                            "'relevancy' จะเน้นข่าวที่เกี่ยวข้องกับคำค้นมากที่สุด เหมาะสำหรับการค้นหาข่าวเฉพาะเจาะจง "
                            "'source' จะจัดเรียงตามแหล่งที่มาของข่าว "
                            "'fetched_at' จะจัดเรียงตามเวลาที่ดึงข้อมูล"
                        )
                    },
                    "datatype": {
                        "type": "string",
                        "enum": ["news", "blog", "multimedia", "forum", "press_release", "review", "research", "opinion", "analysis", "podcast"],
                        "description": (
                            "ประเภทของข้อมูลที่ต้องการดึง 'news' สำหรับข่าวสารทั่วไป และ 'event' สำหรับเหตุการณ์สำคัญหรือกิจกรรมต่างๆ"
                        )
                    }
                },
                # 🔑 จุดสำคัญ: บังคับส่งแค่ language ส่วน query และ category ปล่อยเป็น optional
                "required": ["language"] 
            }
        }
    }
]

# ==========================================
# 4. สร้าง Nodes สำหรับ LangGraph
# ==========================================
def agent_node(state: State):
    """Node หลัก: ประมวลผล ค้นหาซ้ำด้วยคำอื่นหากไม่เจอ และมัดรวมผลลัพธ์จากทุกรอบเป็นข้อๆ"""
    
    messages = state.get("messages")
    if not messages:
        print("🧹 [Stateless Reset] เริ่มต้นค้นหาข่าวสำหรับคำถามใหม่")
        messages = [{"role": "user", "content": state["message"]}]
    
    past_tool_count = sum(1 for m in messages if m.get("role") == "tool")
    print(f"🔄 รอบการค้นหาปัจจุบัน: รอบที่ {past_tool_count + 1}")

    FORMAT_INSTRUCTION = """
⚠️ **กฎเหล็กในการจัดรูปแบบคำตอบ (Strict Formatting Rules):**
ห้ามตอบเป็นข้อความยาวๆ พรืดเดียวกันเด็ดขาด! ให้ใช้ Markdown จัดโครงสร้างให้อ่านง่ายตามรูปแบบนี้เสมอ:

### 🚨 [สรุปเหตุการณ์หลัก หรือ หัวข้อข่าว]
- **📅 วันที่เกิดเหตุ/รายงาน:** (ระบุวันที่/เวลา ถ้ามีข้อมูล)
- **📌 สรุปประเด็นสำคัญ:**
- **💡 Status ล่าสุด:** (สรุปสถานะล่าสุดจากการรวมข้อมูลทุกรอบเข้าด้วยกัน)
- **🔗 แหล่งที่มาของข่าวทั้งหมด:** [คลิกอ่านข่าวเต็มที่นี่](ใส่ URL ลิงก์ทั้งหมดที่เจอในประวัติแชท)

---
"""

    user_raw_text = state["message"]

    if past_tool_count >= 3:
        print("🛑 [Safety Brake] ค้นหาครบ 3 รอบแล้ว บังคับ AI ให้หยุดค้นหาและรวมข้อมูล")
        # 🧠 เพิ่มกฎ "มัดรวมข้อมูล" (Data Aggregation) ในระบบห้ามล้อ
        system_content = f"""คุณคือผู้ช่วยสรุปข่าวสาร 
        ⚠️ ตอนนี้คุณลองค้นหาข่าวมาครบ 3 รอบแล้ว ห้ามเรียกใช้งาน Tool 'fetch_news' เพิ่มเติมเด็ดขาด!
        
        🧠 **หน้าที่ของคุณตอนนี้:**
        ให้คุณย้อนกลับไปอ่านข้อความประเภท 'role': 'tool' **ทั้งหมด** ที่อยู่ในประวัติแชทตั้งแต่รอบแรกจนถึงรอบล่าสุด นำเนื้อหาข่าวสารที่ได้จากทุกคำค้นหามารวมร่าง (Merge) และคัดกรองส่วนที่ซ้ำกันออก เพื่อสร้างเป็นสรุปข่าวที่สมบูรณ์ที่สุดเพียงชุดเดียว ห้ามละทิ้งข้อมูลในรอบแรกๆ เด็ดขาด
        
        {FORMAT_INSTRUCTION}"""
    else:
        # 🏃‍♂️ เพิ่มกฎ "สะสมและมัดรวม" ในรอบการทำงานปกติด้วย เผื่อกรณีที่หาเจอครบตั้งแต่รอบที่ 2
        system_content = f"""คุณคือผู้ช่วยสรุปข่าวสารที่ทำงานตามสั่งอย่างเคร่งครัดและแม่นยำสูง
        
กฎการทำงาน:
1. หากผู้ใช้ถามหาข่าว ให้ใช้ Tool 'fetch_news' เพื่อดึงข้อมูลเสมอ
2. 🔄 **กลยุทธ์การค้นหาหลายรอบ (Multi-turn Search):**
   - ข้อความที่ผู้ใช้พิมพ์มาคือ: "{user_raw_text}"
   - **ขั้นตอนการสร้าง query**:
     a) สกัด keyword สำคัญจากข้อความผู้ใช้ (ชื่อคน สถานที่ เหตุการณ์)
     b) **ใช้ภาษาไทยโดยตรงเสมอ** ห้ามแปลเป็นภาษาอังกฤษเด็ดขาด เช่น "รถไฟ", "น้ำท่วม", "มักกะสัน"
     c) รอบแรกใช้คำที่เฉพาะเจาะจง → ถ้าไม่เจอให้ขยายหรือลองคำทั่วไปกว่า
   - หากดึงข่าวรอบแรกแล้วไม่พบข้อมูล ให้ลองเปลี่ยน query เป็นคำที่กว้างกว่าหรือคำพ้องความหมายภาษาไทย
3. 🧠 **การสรุปผลแบบมัดรวม (Data Aggregation):**
   - นำข้อมูลข่าวสารจากทุกรอบการค้นหามารวมร่างเข้าด้วยกัน ห้ามหยิบมาตอบแค่รอบล่าสุดรอบเดียว
   - สรุปประเด็นสำคัญที่เกี่ยวข้องกับที่ผู้ใช้ต้องการรู้ให้ได้มากที่สุด
4. ✅ **คำค้นหาที่ดี**: ใช้คำภาษาไทย 1-2 คำจากข้อความผู้ใช้ หรือคำพ้องความหมายภาษาไทย
        
        {FORMAT_INSTRUCTION}"""

    local_tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "fetch_news",
                "description": "ใช้สำหรับค้นหาข้อมูลข่าวสารล่าสุดและเหตุการณ์ปัจจุบัน (ข้อมูลย้อนหลัง 48 ชั่วโมง)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                f"คำค้นหาสำคัญที่สกัดจากข้อความผู้ใช้: '{user_raw_text}'\n"
                                "**กฎสำคัญ**: ใช้ภาษาไทยโดยตรง ห้ามแปลเป็นภาษาอังกฤษเด็ดขาด\n"
                                "ตัวอย่าง: ✅ 'น้ำมัน'  ❌ 'oil' | ✅ 'น้ำท่วม'  ❌ 'flood' | ✅ 'รถไฟ'  ❌ 'train'\n"
                                "**บังคับใช้ 1-2 คำเท่านั้น** ห้ามเติมบริบทหรือคำขยายเพิ่มเด็ดขาด:\n"
                                "  ✅ 'น้ำมัน'  ❌ 'ราคาน้ำมันในไทยปรับตัว'\n"
                                "  ✅ 'รถไฟ'  ❌ 'รถไฟชนกันที่มักกะสัน'\n"
                                "  ✅ 'มักกะสัน'  ❌ 'มักกะสันโครงการพัฒนา'\n"
                                "รอบแรกเฉพาะเจาะจง → รอบถัดไปลองคำที่กว้างกว่าหรือคำพ้องความหมาย เช่น 'รถไฟ OR รถด่วน' (นับเป็น 1 query)\n"
                                "⚠️ ถ้า query มีคำมากกว่า 2 คำ (ไม่นับ OR) ให้ตัดออกจนเหลือแค่ 1-2 คำสำคัญ\n"
                                "ถ้าไม่มีคำค้นเฉพาะเจาะจง: ละ parameter นี้ไว้"
                            )
                        },
                        "category": {
                            "type": "string",
                            "enum": ["business", "entertainment", "environment", "food", "health", "politics", "science", "sports", "technology", "world", "top"],
                            "description": "หมวดหมู่ของข่าวสาร ถ้าข่าวทั่วไปให้เลือก 'top' เสมอ"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["th", "en"],
                            "description": "ภาษาของข่าว บังคับใช้ 'th' สำหรับข่าวในประเทศไทย"
                        },
                        "size": {
                            "type": "integer",
                            "description": "จำนวนข่าวที่ต้องการดึง"
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevancy", "source", "fetched_at"]
                        },
                        "datatype": {
                            "type": "string",
                            "enum": ["news", "blog"]
                        }
                    },
                    "required": ["language"] 
                }
            }
        }
    ]

    system_prompt = {"role": "system", "content": system_content}
    full_messages = [system_prompt] + messages
    
    TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")  
    client = OpenAI(base_url='https://api.opentyphoon.ai/v1', api_key=TYPHOON_API_KEY)
    
    print("🤖 [Agent Node] AI กำลังประมวลผลและเตรียมมัดรวมข้อมูล...")
    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct", 
        messages=full_messages,
        tools=local_tools_schema, 
        temperature=0.0 
    )
    
    ai_message = response.choices[0].message
    messages.append(ai_message.model_dump(exclude_none=True))
    
    return {"messages": messages, "final_reply": ai_message.content or ""}

def tool_node(state: State):
    """Node รอง: จะทำงานเมื่อ AI สั่งเรียก Tool เท่านั้น"""
    messages = state["messages"]
    last_ai_message = messages[-1]
    
    if "tool_calls" in last_ai_message:
        for tool_call in last_ai_message["tool_calls"]:
            if tool_call["function"]["name"] == "fetch_news":
                args = json.loads(tool_call["function"]["arguments"])
                news_result = fetch_news_tool(**args)
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "content": news_result
                })
                
    return {"messages": messages}

def router_condition(state: State):
    """ตัวนำทาง: ถ้า AI เรียก Tool ให้ไปที่ Tool Node ถ้าไม่เรียก ให้จบการทำงาน"""
    last_message = state["messages"][-1]
    if "tool_calls" in last_message and last_message["tool_calls"]:
        return "execute_tool"
    return END

# ==========================================
# 5. ประกอบร่าง LangGraph
# ==========================================
workflow = StateGraph(State)

workflow.add_node("agent", agent_node)
workflow.add_node("execute_tool", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router_condition, {"execute_tool": "execute_tool", END: END})
workflow.add_edge("execute_tool", "agent")

agent = workflow.compile()