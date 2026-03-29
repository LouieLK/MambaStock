import uuid
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from src.pipeline import run_mamba_pipeline

app = FastAPI(
    title="MambaQuant API",
    description="基於 Mamba 架構的股市量化預測微服務",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    ts_code: str = Field(..., description="股票代號，例如: 2330.TW", json_schema_extra={"examples": ["2330.TW"]})
    epochs: int = Field(default=50, description="模型訓練次數", ge=1, le=200)
    seq_len: int = Field(default=20, description="用過去幾天預測 (Sliding Window)", ge=5, le=120)
    n_test: int = Field(default=365, description="回測天數", ge=30)

class JobSubmitResponse(BaseModel):
    message: str
    job_id: str

class PredictResult(BaseModel):
    ts_code: str
    next_price: float
    next_pct: float
    status: str

class JobStatusResponse(BaseModel):
    status: str = Field(..., description="任務狀態: pending, processing, completed, failed")
    result: PredictResult | None = Field(default=None, description="預測結果 (若尚未完成則為 null)")
    error: str | None = Field(default=None, description="錯誤訊息 (若未發生錯誤則為 null)")

# 模擬資料庫
jobs_db = {}


def train_and_predict_task(job_id: str, params: dict):
    jobs_db[job_id]["status"] = "processing"
    
    try:
        result = run_mamba_pipeline(params)
        
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["result"] = result
        print(f"[{job_id}] 訓練與預測完成！")
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        print(f"[{job_id}] 任務失敗: {str(e)}")


@app.post("/api/v1/predict", response_model=JobSubmitResponse)
async def create_predict_job(request_data: PredictRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {"status": "pending", "result": None, "error": None}
    
    task_kwargs = request_data.model_dump()
    background_tasks.add_task(train_and_predict_task, job_id, task_kwargs)
    
    return {"message": "任務已建立，模型訓練中...", "job_id": job_id}


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    job = jobs_db.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="找不到此任務 (Job ID not found)")
    
    return job