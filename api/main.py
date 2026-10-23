import io
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.predict import (
    DEFAULT_THRESHOLD,
    load_model,
    predict_batch,
    predict_single,
)

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    try:
        MODEL = load_model()
        print("Model pipeline loaded successfully on startup.")
    except Exception as e:
        print(f"Warning: Model could not be loaded on startup: {e}")
    yield


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready REST API for predicting E-commerce customer churn using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CustomerInput(BaseModel):
    Tenure: float = Field(..., ge=0, description="Tenure with company in months (e.g. 5.0)")
    WarehouseToHome: float = Field(..., ge=0, description="Distance from warehouse to home in km")
    NumberOfDeviceRegistered: int = Field(..., ge=1, le=10, description="Number of devices registered")
    PreferedOrderCat: str = Field(..., description="Laptop & Accessory, Mobile, Fashion, Grocery, Others")
    SatisfactionScore: int = Field(..., ge=1, le=5, description="Customer satisfaction score (1 to 5)")
    MaritalStatus: str = Field(..., description="Single, Married, Divorced")
    NumberOfAddress: int = Field(..., ge=1, description="Number of addresses registered")
    Complain: int = Field(..., ge=0, le=1, description="1 if customer filed a complaint, 0 otherwise")
    DaySinceLastOrder: float = Field(..., ge=0, description="Number of days since last order placed")
    CashbackAmount: float = Field(..., ge=0, description="Average cashback amount received")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Tenure": 2.0,
                "WarehouseToHome": 12.0,
                "NumberOfDeviceRegistered": 3,
                "PreferedOrderCat": "Mobile",
                "SatisfactionScore": 2,
                "MaritalStatus": "Single",
                "NumberOfAddress": 2,
                "Complain": 1,
                "DaySinceLastOrder": 3.0,
                "CashbackAmount": 150.0,
            }
        }
    }


class PredictionOutput(BaseModel):
    churn_prediction: int
    churn_label: str
    churn_probability: float
    risk_level: str
    threshold_used: float


@app.get("/", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, Any]:

    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception:
            pass
    is_ready = MODEL is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "api_version": "1.0.0",
        "model_type": "XGBoost Pipeline",
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(
    customer: CustomerInput,
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold for churn classification"),
) -> PredictionOutput:
    
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    try:
        result = predict_single(customer.model_dump(), model=MODEL, threshold=threshold)
        return PredictionOutput(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_csv(
    file: UploadFile = File(..., description="CSV file containing customer records"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Decision threshold"),
):
    global MODEL
    if MODEL is None:
        try:
            MODEL = load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not loaded: {str(e)}")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        predictions_df = predict_batch(df, model=MODEL, threshold=threshold)

        output_buffer = io.StringIO()
        predictions_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        response = StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=predictions_{file.filename}"},
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True, app_dir=str(PROJECT_ROOT))
