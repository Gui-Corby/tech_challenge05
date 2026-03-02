from fastapi import FastAPI
from predict import router as predict_router

from monitoring.middleware import MetricsMiddleware
from monitoring.routes import router as metrics_router

app = FastAPI(
    title="Passos Mágicos API",
    description="Prever evasão escolar"
)

app.add_middleware(MetricsMiddleware)
app.include_router(metrics_router)


@app.get("/")
async def root():
    return {"message: API is running!"}

app.include_router(predict_router)
