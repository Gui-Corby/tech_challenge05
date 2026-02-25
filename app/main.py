from fastapi import FastAPI

app = FastAPI(
    title="Passos Mágicos API",
    description="Prever evasão escolar"
)

@app.get("/")
async def root():
    return {"message: API is running!"}
