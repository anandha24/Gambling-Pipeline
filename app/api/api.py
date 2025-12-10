from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from tempfile import NamedTemporaryFile
from contextlib import asynccontextmanager
from uuid import uuid4
from datetime import datetime
from PIL import Image
import numpy as np
import shutil
import os
import json
import logging
import time

from app.pipeline.pipeline import GamblingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Initialize pipeline 
pipeline = GamblingPipeline()


def warmup_pipeline(pipeline_instance: GamblingPipeline):
    logger.info("[WARMUP] Starting model warmup...")
    warmup_start = time.time()
    
    try:
        t_start = time.time()
        dummy_image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        _ = pipeline_instance.classifier.predict_prob(dummy_image)
        logger.info(f"[WARMUP] ViT Classifier ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        t_start = time.time()
        dummy_image_det = Image.new("RGB", (640, 640), color=(128, 128, 128))
        _ = pipeline_instance.detector.detect(dummy_image_det)
        logger.info(f"[WARMUP] RT-DETR Detector ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        t_start = time.time()
        dummy_array = np.zeros((100, 300, 3), dtype=np.uint8)
        dummy_array.fill(200)
        _ = pipeline_instance.ocr.reader.readtext(dummy_array, detail=0)
        logger.info(f"[WARMUP] EasyOCR ready ({(time.time() - t_start) * 1000:.0f}ms)")
        
        total_warmup_time = (time.time() - warmup_start) * 1000
        logger.info(f"[WARMUP] All models warmed up ({total_warmup_time:.0f}ms)")
        
    except Exception as e:
        logger.error(f"[WARMUP] Warmup failed: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Gambling Detection API...")
    warmup_pipeline(pipeline)
    logger.info("Server is ready to accept requests")
    
    yield
    
    logger.info("Shutting down server...")


app = FastAPI(
    title="Gambling Detection API",
    description="Classification → Object Detection → OCR",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("results/inference", exist_ok=True)
os.makedirs("results/data", exist_ok=True)

app.mount("/results", StaticFiles(directory="results"), name="results")


@app.get("/health")
async def health_check():
    """Health check endpoint untuk monitoring API status"""
    return JSONResponse({
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    })

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name

        result = pipeline.process(temp_path)

        # Logging performance
        perf = result.get("performance", {})
        logger.info(
            f"[PREDICT] Status: {result['status']} | "
            f"Total: {perf.get('total_ms', 0)}ms | "
            f"Classifier: {perf.get('classifier_ms', 0)}ms | "
            f"OCR: {perf.get('ocr_ms', 0)}ms | "
            f"Detector: {perf.get('detector_ms', 0)}ms | "
            f"Viz: {perf.get('visualization_ms', 0)}ms"
        )

        result_id = str(uuid4())
        result["id"] = result_id
        result["timestamp"] = datetime.now().isoformat()

        json_path = f"results/data/{result_id}.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=4)

        return JSONResponse({"success": True, "result": result})

    except Exception as e:
        logger.error(f"[PREDICT] Error: {str(e)}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.get("/results")
async def get_all_results():
    results = []

    for filename in os.listdir("results/data"):
        if filename.endswith(".json"):
            with open(os.path.join("results/data", filename)) as f:
                results.append(json.load(f))
    
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return JSONResponse({"success": True, "results": results})

@app.get("/result/{result_id}")
async def get_result_detail(result_id: str):
    json_path = f"results/data/{result_id}.json"
    if not os.path.exists(json_path):
        return JSONResponse({"success": False, "message": "Result not found."}, status_code=404)

    with open(json_path) as f:
        data = json.load(f)

    return JSONResponse({"success": True, "result": data})
