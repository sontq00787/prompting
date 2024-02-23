from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import bittensor as bt
# Bittensor Miner Template:
import prompting
from prompting.llm import load_pipeline, HuggingFaceLLM

app = FastAPI()

# Load the HuggingFace model
# model_kwargs: dict = None
model_id = "HuggingFaceH4/zephyr-7b-beta"  # Example model ID
llm_pipeline = load_pipeline(model_id=model_id, torch_dtype=torch.float16, device="cuda" if torch.cuda.is_available() else "cpu")


class QueryInput(BaseModel):
    prompt: str

    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(query_input: QueryInput):
    bt.logging.debug(f"Received query: {query_input}")
        # --neuron.max_tokens 64 --neuron.do_sample True --neuron.temperature 0.9 --neuron.top_k 50 --neuron.top_p 0.95
    completion = HuggingFaceLLM(
        llm_pipeline=llm_pipeline,
        system_prompt="You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know.",
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
    ).query(
        message=query_input.prompt,
        role="user",
        disregard_system_prompt=False,
    )
    return {"completion": completion}