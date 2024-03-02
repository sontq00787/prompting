import uvicorn

port = 8868 # This is the port number where the FastAPI server will run, you can change it to any other port number
if __name__ == "__main__":
    uvicorn.run("mistral_api:app", host="0.0.0.0", port=port, workers=2)