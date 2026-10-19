import base64
from dotenv import load_dotenv
load_dotenv()
import os
import shutil
import uuid
import json
import requests
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

app = FastAPI()

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据库连接配置（SQLite）
SQLALCHEMY_DATABASE_URL = "sqlite:///./agri_disease.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Farmland(Base):
    __tablename__ = "farmlands"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    crop = Column(String(50))
    area = Column(Float)
    growth_stage = Column(String(20))


class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True, index=True)
    farmland_id = Column(Integer)
    status = Column(String(20), default="待检测")
    create_time = Column(String(50), default=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(Integer)
    image_url = Column(Text)
    disease_type = Column(String(50))
    severity = Column(String(20))
    confidence = Column(Float)
    bbox_json = Column(Text)
    heatmap_url = Column(String(255))
    heatmap_data = Column(Text)  # 存网格热力图数据
    review_status = Column(String(20), default="AI_PENDING")


class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    id = Column(Integer, primary_key=True, index=True)
    disease_type = Column(String(50))
    severity = Column(String(20))
    suggestion = Column(Text)
    pesticide_recommendation = Column(String(255))


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    task_id = Column(Integer)
    message = Column(String(255))
    is_read = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 根路径测试
@app.get("/")
def read_root():
    return {"message": "后端服务运行正常"}


# 地块管理
@app.post("/api/farmlands")
def create_farmland(name: str, crop: str, area: float, growth_stage: str, db: Session = Depends(get_db)):
    new_farmland = Farmland(name=name, crop=crop, area=area, growth_stage=growth_stage)
    db.add(new_farmland)
    db.commit()
    db.refresh(new_farmland)
    return {"code": 200, "message": "建档成功", "data": {"farmland_id": new_farmland.id}}


@app.get("/api/farmlands")
def get_farmlands(db: Session = Depends(get_db)):
    farmlands = db.query(Farmland).all()
    data = [{"id": f.id, "name": f.name, "crop": f.crop, "area": f.area, "growth_stage": f.growth_stage} for f in farmlands]
    return {"code": 200, "message": "success", "data": data}


# 巡田任务
@app.post("/api/tasks")
def create_task(farmland_id: int, db: Session = Depends(get_db)):
    new_task = Task(farmland_id=farmland_id)
    db.add(new_task)
    db.commit()
    db.refresh(new_task)
    return {"code": 200, "message": "success", "data": {"task_id": new_task.id}}


@app.get("/api/tasks")
def get_tasks(db: Session = Depends(get_db)):
    tasks = db.query(Task).all()
    data = [{"task_id": t.id, "farmland_id": t.farmland_id, "status": t.status} for t in tasks]
    return {"code": 200, "message": "success", "data": data}


# 图片上传
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{image_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"code": 200, "message": "上传成功",
            "data": {"image_id": image_id, "image_url": f"http://127.0.0.1:8001/uploads/{filename}"}}


# AI手动录入检测结果（备用）
@app.post("/api/detect/result")
def save_detect_result(task_id: int, disease_type: str, severity: str, confidence: float, bbox_list: str,
                       heatmap_url: str, db: Session = Depends(get_db)):
    new_record = Detection(task_id=task_id, disease_type=disease_type, severity=severity, confidence=confidence,
                           bbox_json=bbox_list, heatmap_url=heatmap_url)
    db.add(new_record)
    task = db.query(Task).filter(Task.id == task_id).first()
    if task:
        task.status = "已完成"
    db.commit()
    return {"code": 200, "message": "检测结果已保存", "data": {"record_id": new_record.id}}


# 病害统计
@app.get("/api/stats/{farmland_id}")
def get_statistics(farmland_id: int, db: Session = Depends(get_db)):
    tasks = db.query(Task).filter(Task.farmland_id == farmland_id).all()
    task_ids = [t.id for t in tasks]
    records = db.query(Detection).filter(Detection.task_id.in_(task_ids)).all() if task_ids else []

    total_count = len(records)
    severe_count = len([r for r in records if r.severity == "重"])
    incidence_rate = round((total_count / 100.0), 2) if total_count else 0.0
    disease_index = round((severe_count * 0.8 + (total_count - severe_count) * 0.3) / total_count,
                          2) if total_count else 0.0

    return {"code": 200, "message": "success",
            "data": {"total_detected": total_count, "severe_count": severe_count, "incidence_rate": incidence_rate,
                     "disease_index": disease_index}}

# 防治建议
@app.get("/api/suggestions/{task_id}")
def get_suggestions(task_id: int, db: Session = Depends(get_db)):
    record = db.query(Detection).filter(Detection.task_id == task_id).first()
    if not record:
        return {"code": 404, "message": "未找到检测记录", "data": None}

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.disease_type == record.disease_type,
                                        KnowledgeBase.severity == record.severity).first()

    if not kb:
        return {"code": 200, "message": "success", "data": {"suggestion": "暂无建议，请咨询植保站。", "pesticide": "无"}}

    return {"code": 200, "message": "success",
            "data": {"suggestion": kb.suggestion, "pesticide": kb.pesticide_recommendation,
                     "prescription_map_url": record.heatmap_url}}

# 专家复核
@app.post("/api/expert/review")
def expert_review(record_id: int, corrected_disease: str, corrected_severity: str, db: Session = Depends(get_db)):
    record = db.query(Detection).filter(Detection.id == record_id).first()
    if not record:
        return {"code": 404, "message": "记录不存在", "data": None}

    record.disease_type = corrected_disease
    record.severity = corrected_severity
    record.review_status = "PUBLISHED"

    new_notification = Notification(user_id=1, task_id=record.task_id,
                                    message=f"您的任务{record.task_id}检测报告已由专家复核完毕。")
    db.add(new_notification)
    db.commit()
    return {"code": 200, "message": "复核发布成功，已生成通知", "data": None}

# 消息通知
@app.get("/api/notifications")
def get_notifications(db: Session = Depends(get_db)):
    notifications = db.query(Notification).filter(Notification.is_read == 0).all()
    data = [{"id": n.id, "message": n.message, "task_id": n.task_id} for n in notifications]
    return {"code": 200, "message": "success", "data": data}

# 微信登录
WECHAT_APPID = "wxcec06bd690fa435f"
WECHAT_SECRET = "71ba26e79556d84597fbdfcb3af8b393"

@app.post("/api/login")
def login(code: str = Body(..., embed=True), db: Session = Depends(get_db)):
    if code in ["farmer", "expert"]:
        role = "expert" if code == "expert" else "farmer"
        username = "植保站王专家" if role == "expert" else "东区张老汉"
        return {"code": 200, "message": "ok",
                "data": {"token": f"token_{uuid.uuid4().hex[:8]}", "role": role, "user_id": 1, "username": username}}

    url = f"https://api.weixin.qq.com/sns/jscode2session?appid={WECHAT_APPID}&secret={WECHAT_SECRET}&js_code={code}&grant_type=authorization_code"
    try:
        res = requests.get(url, timeout=5).json()
        if res.get("errcode"):
            print("wx err:", res)
            raise Exception()
        return {"code": 200, "message": "ok",
                "data": {"token": f"token_{uuid.uuid4().hex[:8]}", "openid": res.get("openid"),
                         "session_key": res.get("session_key"), "role": "farmer", "username": "东区张老汉"}}
    except Exception as e:
        print("wx fail:", e)
        return {"code": 200, "message": "ok",
                "data": {"token": f"token_{uuid.uuid4().hex[:8]}", "openid": "mock_openid",
                         "session_key": "mock_key", "role": "farmer", "username": "东区张老汉"}}

# 发送短信验证码
@app.post("/api/sms/send")
def send_sms(phone: str = Body(..., embed=True)):
    print(f"[模拟发送短信] 手机号: {phone}, 验证码固定为: 123456")
    return {
        "code": 200,
        "message": "验证码已发送（演示模拟，请输入123456）",
        "data": {"sms_code": "123456"}
    }

# 手机号验证码登录
@app.post("/api/sms/login")
def sms_login(phone: str = Body(..., embed=True), sms_code: str = Body(..., embed=True), db: Session = Depends(get_db)):
    if sms_code != "123456":
        return {"code": 400, "message": "验证码错误，请重新输入（演示版请输入123456）", "data": None}

    return {
        "code": 200,
        "message": "登录成功",
        "data": {
            "token": f"token_sms_{uuid.uuid4().hex[:8]}",
            "role": "farmer",
            "user_id": 2,
            "username": f"农户_{phone[-4:]}"
        }
    }


# 历史记录
@app.get("/api/history")
def get_history(farmland_id: int = None, db: Session = Depends(get_db)):
    if farmland_id is not None:
        tasks = db.query(Task).filter(Task.farmland_id == farmland_id).all()
        task_ids = [t.id for t in tasks]
        records = db.query(Detection).filter(Detection.task_id.in_(task_ids)).order_by(Detection.id.desc()).all()
    else:
        records = db.query(Detection).order_by(Detection.id.desc()).all()

    data = []
    for r in records:
        task = db.query(Task).filter(Task.id == r.task_id).first()
        f_id = task.farmland_id if task else None

        data.append({
            "record_id": r.id,
            "task_id": r.task_id,
            "farmland_id": f_id,
            "disease_type": r.disease_type,
            "severity": r.severity,
            "confidence": r.confidence,
            "review_status": r.review_status,
            "image_url": r.image_url
        })
    return {"code": 200, "message": "success", "data": data}

# 结果详情
@app.get("/api/detect/result/{id_or_task_id}")
def get_detect_result(id_or_task_id: int, db: Session = Depends(get_db)):
    record = db.query(Detection).filter(Detection.task_id == id_or_task_id).first()
    if not record:
        record = db.query(Detection).filter(Detection.id == id_or_task_id).first()

    if not record:
        return {"code": 404, "message": "未找到记录", "data": None}

    return {
        "code": 200,
        "message": "success",
        "data": {
            "task_id": record.task_id,
            "record_id": record.id,
            "disease_type": record.disease_type,
            "severity": record.severity,
            "confidence": record.confidence,
            "image_url": record.image_url,
            "bbox_list": json.loads(record.bbox_json) if record.bbox_json else [],
            "heatmap_data": json.loads(record.heatmap_data) if record.heatmap_data else {},  # 热力图数据
            "review_status": record.review_status
        }
    }

# 一级：上传航拍图
@app.post("/api/algo/detect_drone")
async def algo_detect_drone(task_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = await file.read()

    image_id = f"img_{uuid.uuid4().hex[:8]}"
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{image_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)
    image_url = f"https://enduring-unmade-statue.ngrok-free.dev/uploads/{filename}"
    files = {'file': (file.filename, file_bytes, file.content_type)}

    try:
        algo_url = "http://127.0.0.1:8002/algo/predict"
        res = requests.post(algo_url, files=files, timeout=15).json()

        if res.get("code") != 200:
            return {"code": 500, "message": "航拍算法报错", "data": None}

        disease_cn = res.get("disease_name_cn", "未知")
        severity = res.get("severity", "unknown")
        confidence = res.get("confidence", 0.0)
        boxes = res.get("boxes", [])
        advice = res.get("advice", {})
        statistics = res.get("statistics", {})

        new_record = Detection(
            task_id=task_id,
            image_url=image_url,
            disease_type=disease_cn,
            severity=severity,
            confidence=confidence,
            bbox_json=json.dumps(boxes),
            heatmap_data=json.dumps(statistics)
        )
        db.add(new_record)
        task = db.query(Task).filter(Task.id == task_id).first()
        if task: task.status = "已完成"
        db.commit()
        db.refresh(new_record)

        return {"code": 200, "message": "航拍检测成功", "data": {
            "record_id": new_record.id,
            "image_url": image_url,
            "disease_type": disease_cn, "severity": severity,
            "confidence": confidence, "bbox_list": boxes, "advice": advice,
            "need_expert_review": advice.get("need_expert_review", False)
        }}
    except Exception as e:
        print(f"航拍接口挂了: {e}")
        new_record = Detection(task_id=task_id, image_url=image_url, disease_type="受胁迫", severity="severe",
                               confidence=0.5, bbox_json=json.dumps([]), heatmap_data=json.dumps({}))
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        return {"code": 200, "message": "算法超时（兜底）",
                "data": {"record_id": new_record.id, "image_url": image_url, "disease_type": "受胁迫",
                         "severity": "severe", "confidence": 0.5, "bbox_list": [],
                         "need_expert_review": True}}


# 二级：手机上传
@app.post("/api/algo/detect_phone")
async def algo_detect_phone(task_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    file_bytes = await file.read()

    # ！先保存图片到本地
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"{image_id}{file_extension}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as buffer:
        buffer.write(file_bytes)
        image_url = f"https://enduring-unmade-statue.ngrok-free.dev/uploads/{filename}"
    files = {'file': (file.filename, file_bytes, file.content_type)}

    try:
        algo_url = "http://127.0.0.1:8003/algo/classify"
        res = requests.post(algo_url, files=files, timeout=15).json()

        if res.get("code") != 200:
            return {"code": 500, "message": "手机识别算法报错", "data": None}

        disease_cn = res.get("disease_name_cn", "未知")
        severity = res.get("severity", "unknown")
        confidence = res.get("confidence", 0.0)
        boxes = res.get("boxes", [])
        advice = res.get("advice", {})
        statistics = res.get("statistics", {})

        new_record = Detection(
            task_id=task_id,
            image_url=image_url,
            disease_type=disease_cn,
            severity=severity,
            confidence=confidence,
            bbox_json=json.dumps(boxes),
            heatmap_data=json.dumps(statistics)
        )
        db.add(new_record)
        task = db.query(Task).filter(Task.id == task_id).first()
        if task: task.status = "已完成"
        db.commit()
        db.refresh(new_record)

        return {"code": 200, "message": "手机识别成功", "data": {
            "record_id": new_record.id,
            "image_url": image_url,
            "disease_type": disease_cn, "severity": severity,
            "confidence": confidence, "bbox_list": boxes, "advice": advice,
            "need_expert_review": advice.get("need_expert_review", False)
        }}
    except Exception as e:
        print(f"手机接口挂了: {e}")
        new_record = Detection(task_id=task_id, image_url=image_url, disease_type="受胁迫", severity="severe",
                               confidence=0.5, bbox_json=json.dumps([]), heatmap_data=json.dumps({}))
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        return {"code": 200, "message": "算法超时（兜底）",
                "data": {"record_id": new_record.id, "image_url": image_url, "disease_type": "受胁迫",
                         "severity": "severe", "confidence": 0.5, "bbox_list": [],
                         "need_expert_review": True}}
#通义千问：个性化
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://ws-5vbmof6mxst4l92a.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

class AdviceRequest(BaseModel):
    task_id: int

@app.post("/api/chat/advice")
def get_personalized_advice(request: AdviceRequest, db: Session = Depends(get_db)):
    record = db.query(Detection).filter(Detection.task_id == request.task_id).first()
    if not record:
        return {"code": 404, "message": "未找到检测记录", "data": None}

    disease = record.disease_type
    severity = record.severity
    confidence = record.confidence

    prompt = f"""
    我是一个农民，我的农田刚刚被无人机/手机AI检测系统扫描过了。
    检测结果显示：
    - 病害类型：{disease}
    - 严重程度：{severity}
    - AI置信度：{confidence}

    请你作为专业的中国农业植保专家，针对上面这个具体的检测结果，给我一段通俗易懂、可操作性强的防治建议。
    要求：包含立即要做的农事操作、打药建议（不要推荐具体违禁农药），以及后续的预防工作。
    """
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "你是一个专业的中国农业植保专家，请用通俗易懂的中文回答。"},
                {"role": "user", "content": prompt}
            ],
            timeout=15
        )
        answer = response.choices[0].message.content
        return {"code": 200, "message": "success", "data": {"advice": answer}}

    except Exception as e:
        print(f"通义千问调用失败: {e}")
        return {
            "code": 200,
            "message": "success",
            "data": {"advice": f"针对您的 {disease} ({severity})，建议立即清除病残体，保持通风，并咨询当地植保站用药。"}
        }