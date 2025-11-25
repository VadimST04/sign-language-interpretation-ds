import os
import uuid
import shutil
import json
import asyncio

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from pipeline import SubtitlePipeline

app = FastAPI(title="Unified Subtitle API")

DEFAULT_STYLE = {
    "fontSize": 28,
    "font": "Arial",
    "color": "#FFFFFF"
}

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
async def create_subtitled_video(
    video: UploadFile = File(...),
    lang: str = Form(...),

    fontSize: int = Form(28),
    font: str = Form("Arial"),
    color: str = Form("#FFFFFF"),
):
    """
    Swagger will show distinct form fields:
    - video
    - lang
    - fontSize
    - font
    - color
    """

    # -------- Validate HEX color --------
    if not (isinstance(color, str) and color.startswith("#") and len(color) == 7):
        raise HTTPException(400, "Color must be HEX '#RRGGBB' format")

    supported = ["uk", "pl", "es", "fr", "de", "it", "ru"]
    if lang not in supported:
        raise HTTPException(400, f"Unsupported lang={lang}")

    # -------------- TEMP FOLDER --------------
    temp_id = uuid.uuid4().hex
    folder = f"temp/{temp_id}"
    os.makedirs(folder, exist_ok=True)

    input_path = f"{folder}/input.mp4"
    output_path = f"{folder}/output.mp4"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # -------- Run Pipeline with styling --------
    pipeline = get_pipeline(lang)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        lambda: pipeline.process(
            video_path=input_path,
            output_path=output_path,
            font_size=fontSize,
            font_color=color,
            font_family=font
        )
    )

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"video_{lang}_styled.mp4"
    )


@app.get("/")
def health():
    return {"status": "running"}
