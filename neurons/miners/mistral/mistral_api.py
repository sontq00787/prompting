from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import bittensor as bt
# Bittensor Miner Template:
# from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

app = FastAPI()

# Load the HuggingFace model
# model_kwargs: dict = None
model_id = "Open-Orca/Mistral-7B-OpenOrca"  # Example model ID

# Load model
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16 )
tokenizer = AutoTokenizer.from_pretrained(model_id)


class QueryInput(BaseModel):
    prompt: str

    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(query_input: QueryInput):
    bt.logging.debug(f"Received query: {query_input}")
        # --neuron.max_tokens 64 --neuron.do_sample True --neuron.temperature 0.9 --neuron.top_k 50 --neuron.top_p 0.95
    
    system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
    prompt = query_input.prompt
    prompt_template =  f'''<|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant

        '''

    print("\n\n*** Generate:")

    tokens = tokenizer(
        prompt_template,
        return_tensors='pt'
    ).input_ids.cuda()

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=1024
    )

    # print("Output: ", tokenizer.decode(generation_output[0]))
    completion = extract_assistant_response(tokenizer.decode(generation_output[0]))

    # print("*** Pipeline:")
    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=1024,
    #     do_sample=True,
    #     temperature=0.7,
    #     top_p=0.95,
    #     top_k=40,
    #     repetition_penalty=1.1
    # )

    # completion =  pipe(prompt_template)[0]['generated_text']

    # completion = HuggingFaceLLM(
    #     llm_pipeline=llm_pipeline,
    #     system_prompt="You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know.",
    #     max_new_tokens=1024,
    #     do_sample=True,
    #     temperature=0.9,
    #     top_k=50,
    #     top_p=0.95,
    # ).query(
    #     message=query_input.prompt,
    #     role="user",
    #     disregard_system_prompt=False,
    # )
    torch.cuda.empty_cache()
    return {"completion": completion}


def extract_assistant_response(completion: str) -> str:
    # Splitting the completion string at "assistant\n\n"
    parts = completion.split("assistant\n\n")
    
    # The assistant's response is expected to be after the last occurrence of "assistant\n\n"
    if len(parts) > 1:
        # Taking everything after "assistant\n\n" and stripping leading/trailing whitespace
        assistant_response = parts[-1].strip()
    else:
        # If "assistant\n\n" is not found, return an indication that the response couldn't be extracted
        assistant_response = "I'm sorry, I couldn't extract the response."
    
    return assistant_response