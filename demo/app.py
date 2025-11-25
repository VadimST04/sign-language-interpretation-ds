import os
import uuid
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from pipeline import SubtitlePipeline

app = FastAPI(title="Subtitle API")

pipelines = {}


def get_pipeline(lang: str):
    if lang not in pipelines:
        model = f"Helsinki-NLP/opus-mt-en-{lang}"
        pipelines[lang] = SubtitlePipeline(
            whisper_model="medium",
            translator_model=model
        )
    return pipelines[lang]


@app.post("/subtitles")
async def subtitles(video: UploadFile = File(...), lang: str = Form("uk")):

    supported = ["uk", "pl", "es", "fr", "de", "it", "ru"]
    if lang not in supported:
        raise HTTPException(400, f"Unsupported lang={lang}")

    temp_id = str(uuid.uuid4())
    os.makedirs(f"temp/{temp_id}", exist_ok=True)

    input_path = f"temp/{temp_id}/input.mp4"
    output_path = f"temp/{temp_id}/output.mp4"

    # Save file
    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    pipeline = get_pipeline(lang)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: pipeline.process(input_path, output_path))

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"video_{lang}_subtitles.mp4"
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Subtitle server running"}
