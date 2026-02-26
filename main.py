import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#API key
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")

client = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
)


@app.get("/")
async def root():
    return {"message": "AI 简历分析系统后端运转正常！"}

@app.post("/api/analyze")
async def analyze_resume(
    file: UploadFile = File(...),
    job_desc: str = Form("无特定岗位要求") 
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="只能上传 PDF 格式的简历哦！")
    
    try:
        resume_text = extract_text(file.file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF 读取失败: {str(e)}")
        
    if len(resume_text) < 20:
        raise HTTPException(status_code=400, detail="简历字数太少，无法分析。")

    prompt = f"""
    你是一个专业的 HR 助手。请分析以下简历，并根据岗位需求打分。
    
    【岗位需求】：{job_desc}
    【简历内容】：{resume_text}
    
    请直接返回 JSON 格式数据，包含这三个固定字段：
    1. basic_info (姓名, 电话, 邮箱)
    2. extra_info (学历, 核心技能概括)
    3. evaluation (匹配度0-100的数字, 结合岗位需求给出具体的评分理由)
    
    只输出 JSON，不要有任何多余的废话。
    """

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}]
        )
        result_str = response.choices[0].message.content.strip()
        
        if result_str.startswith("```json"):
            result_str = result_str[7:-3]
            
        return json.loads(result_str)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 分析失败: {str(e)}")