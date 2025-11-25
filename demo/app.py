import os
import uuid
import shutil
import json
import asyncio
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import SubtitlePipeline

app = FastAPI(title="Subtitle Generator API")


SETTINGS_FILE = "subtitle_settings.json"

DEFAULT_SETTINGS = {
    "font_size": 28,
    "font_color": "#FFFFFF",
    "font_family": "Arial"
}

if not os.path.exists(SETTINGS_FILE):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_SETTINGS, f, indent=4)


def load_subtitle_settings():
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_subtitle_settings(data: dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


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
        raise HTTPException(400, f"Unsupported language: {lang}")

    settings = load_subtitle_settings()

    temp_id = uuid.uuid4().hex
    folder = f"temp/{temp_id}"
    os.makedirs(folder, exist_ok=True)

    input_path = f"{folder}/input.mp4"
    output_path = f"{folder}/output.mp4"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    pipeline = get_pipeline(lang)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: pipeline.process(
            video_path=input_path,
            output_path=output_path,
            font_size=settings["font_size"],
            font_color=settings["font_color"],
            font_family=settings["font_family"]
        )
    )

    return FileResponse(output_path, media_type="video/mp4")


class SubtitleSettings(BaseModel):
    font_size: Optional[int] = None
    font_color: Optional[str] = None   # HEX
    font_family: Optional[str] = None


@app.post("/settings/subtitles")
def update_settings(settings: SubtitleSettings):

    current = load_subtitle_settings()
    updates = settings.dict(exclude_unset=True)

    for key, value in updates.items():
        current[key] = value

    save_subtitle_settings(current)

    return {
        "status": "ok",
        "new_settings": current
    }


@app.get("/settings/subtitles")
def get_current():
    return load_subtitle_settings()


@app.get("/")
def root():
    return {"status": "running"}
