import os
from typing import List
from contextlib import asynccontextmanager
from comet import download_model, load_from_checkpoint
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

model = None

API_KEY = os.getenv("CONF_API_KEY", "")
HOSTNAME = os.getenv("HOSTNAME", "")
PORT = os.getenv("PORT_QE", "")

class RewardsRequest(BaseModel):
    completions: List[str] = Field(..., min_length=1)
    mt: List[str] = Field(..., min_length=1)

class RewardsResponse(BaseModel):
    qe_rewards: List[float]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    # Load model once
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    yield


app = FastAPI(lifespan=lifespan)

@app.post("/qe-rewards", response_model=RewardsResponse)
def qe_rewards(req: RewardsRequest, x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if len(req.completions) != len(req.mt):
        raise HTTPException(status_code=400, detail="completions and subjects must be same length")

    data = []
    for src, mt in zip(req.completions, req.mt):
        data.append({
            "src": src,
            "mt": mt
        })
    
    scores = model.predict(data, batch_size=8, gpus=1)['scores']
    return RewardsResponse(qe_rewards=[float(x) for x in scores])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service_da:app", host=HOSTNAME, port=PORT)