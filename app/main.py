from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.predict import router as predict_router
from app.routes import router as monitoring_router

from app.monitoring.middleware import MetricsMiddleware


app = FastAPI(
    title="Passos Mágicos API",
    description="Prever evasão escolar"
)

app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "API is running!"}

app.include_router(predict_router)
app.include_router(monitoring_router)

