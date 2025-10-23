import torch
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import FileResponse
import soundfile as sf
import tempfile, os
import nest_asyncio
import uvicorn
from pyngrok import ngrok
import os
from google.colab import userdata

# Load Riffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("riffusion/riffusion-model-v1", torch_dtype=torch.float16)
pipe = pipe.to(device)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Music Composer is running!"}

@app.get("/compose")
def compose(prompt: str = "lofi hip hop beats"):
    # Generate audio from text prompt
    result = pipe(prompt, num_inference_steps=50)
    audio = result.audios[0]  # numpy array

    # Save to temporary WAV file
    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sf.write(tmpfile.name, audio, 44100)

    return FileResponse(tmpfile.name, media_type="audio/wav", filename="riffusion.wav")


# Allow nested event loops in Colab
nest_asyncio.apply()

# Set ngrok authtoken from Colab secrets
ngrok_auth_token = userdata.get('NGROK_AUTH_TOKEN')
if ngrok_auth_token:
    print("NGROK_AUTH_TOKEN successfully retrieved from secrets.")
    ngrok.set_auth_token(ngrok_auth_token)
else:
    print("NGROK_AUTH_TOKEN secret not found. Please add it to Colab secrets.")


# Open a ngrok tunnel to the API
public_url = ngrok.connect(8000).public_url
print("Public FastAPI URL:", public_url)

# Run FastAPI app
uvicorn.run(app, host="0.0.0.0", port=8000)
