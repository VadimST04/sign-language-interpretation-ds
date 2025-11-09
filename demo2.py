"""
Automatic English → Ukrainian Subtitles Generator

Pipeline:
1. Extract audio from video.
2. Transcribe English speech with Whisper.
3. Translate recognized English text into Ukrainian.
4. Generate Ukrainian subtitles (.srt).
5. Burn translated subtitles into the video using FFmpeg.
"""

import os
import subprocess
from typing import Dict, List

import torch
import whisper
from transformers import pipeline as hf_pipeline


class AudioExtractor:
    """Extracts audio from a video file using FFmpeg."""

    def extract(self, video_path: str, output_audio: str = "temp_audio.wav") -> str:
        """Extract audio (16kHz mono WAV) from a given video."""
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", output_audio
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Аудіо витягнуто: {output_audio}")
        return output_audio


class SpeechRecognizer:
    """Performs English speech recognition using Whisper."""

    def __init__(self, model_size: str = "medium", device: str = None):
        """Initialize Whisper model."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Використовується пристрій для Whisper: {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_path: str) -> Dict:
        """
        Transcribe English audio into text with segment timestamps.
        """
        result = self.model.transcribe(
            audio_path,
            language="en",
            word_timestamps=False
        )
        print(f"✓ Розпізнано англійський текст: {result['text'][:100]}...")
        return result


class Translator:
    """Translates English text into Ukrainian using a pretrained transformer model."""

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-uk"):
        """
        Initialize the translation model.

        Args:
            model_name: Name of the translation model from Hugging Face.
        """
        print(f"Завантаження моделі перекладу: {model_name}")
        self.translator = hf_pipeline("translation", model=model_name)
        print("✓ Модель перекладу завантажено")

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Translate each segment's text from English to Ukrainian.

        Args:
            segments: List of transcription segments from Whisper.

        Returns:
            List of segments with translated 'text' fields.
        """
        translated_segments = []
        for seg in segments:
            en_text = seg.get("text", "").strip()
            if not en_text:
                continue
            try:
                translated_text = self.translator(en_text)[0]["translation_text"]
            except Exception as e:
                translated_text = en_text  # fallback
                print(f"⚠️ Помилка перекладу сегмента: {e}")

            translated_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": translated_text
            })

        print(f"✓ Перекладено сегментів: {len(translated_segments)}")
        return translated_segments


class SubtitleGenerator:
    """Creates an SRT subtitle file from translated segments."""

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

    def generate_srt(self, segments: List[Dict], output_path: str = "subtitles.srt") -> str:
        """Generate an SRT subtitle file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = self._format_timestamp(seg["start"])
                end = self._format_timestamp(seg["end"])
                text = seg["text"].strip().replace("\n", " ")
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
        print(f"✓ Субтитри створено: {output_path}")
        return output_path


class VideoSubtitler:
    """Burns subtitles into a video using FFmpeg."""

    @staticmethod
    def _escape_path(path: str) -> str:
        """Escape Windows path for FFmpeg subtitles filter."""
        return path.replace("\\", "\\\\").replace(":", "\\:")

    def burn_subtitles(self, input_video: str, srt_path: str, output_video: str) -> str:
        """
        Burn subtitles permanently into a video.
        """
        srt_escaped = self._escape_path(srt_path)
        vf = f"subtitles='{srt_escaped}':force_style='Fontsize=28,Outline=2,Shadow=1,MarginV=25'"

        cmd = [
            "ffmpeg", "-i", input_video,
            "-vf", vf,
            "-c:a", "copy",
            "-y", output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Субтитри впалено у відео: {output_video}")
        return output_video


class EnglishToUkrainianSubtitlePipeline:
    """
    Full pipeline:
        1. Extract audio from video
        2. Transcribe English speech with Whisper
        3. Translate text into Ukrainian
        4. Generate SRT file
        5. Burn subtitles into video
    """

    def __init__(self, whisper_model: str = "medium", translator_model: str = "Helsinki-NLP/opus-mt-en-uk"):
        self.audio_extractor = AudioExtractor()
        self.speech_recognizer = SpeechRecognizer(whisper_model)
        self.translator = Translator(translator_model)
        self.subtitle_generator = SubtitleGenerator()
        self.video_subtitler = VideoSubtitler()

    def process(self, video_path: str, output_path: str = "video_with_ukrainian_subtitles.mp4") -> str:
        """Run the full English→Ukrainian subtitle generation process."""
        print("=" * 70)
        print("🚀 ЗАПУСК PIPELINE: АНГЛІЙСЬКІ → УКРАЇНСЬКІ СУБТИТРИ")
        print("=" * 70)

        # 1. Extract audio
        audio_path = self.audio_extractor.extract(video_path)

        # 2. Transcribe English
        transcription = self.speech_recognizer.transcribe(audio_path)

        # 3. Translate to Ukrainian
        translated_segments = self.translator.translate_segments(transcription["segments"])

        # 4. Create Ukrainian SRT
        srt_path = self.subtitle_generator.generate_srt(translated_segments, "temp_subtitles.srt")

        # 5. Burn subtitles into video
        self.video_subtitler.burn_subtitles(video_path, srt_path, output_path)

        # Cleanup
        try:
            os.remove(audio_path)
            os.remove(srt_path)
        except OSError:
            pass

        print("=" * 70)
        print(f"✅ ГОТОВО! Відео з українськими субтитрами: {output_path}")
        print("=" * 70)
        return output_path


if __name__ == "__main__":
    pipeline = EnglishToUkrainianSubtitlePipeline(whisper_model="medium")

    input_video = r"D:\PyCharm\PyProjects\IntPreparation\ds-interpretation\input_video.mp4"
    output_video = "output_with_ukrainian_subtitles.mp4"

    result = pipeline.process(video_path=input_video, output_path=output_video)
    print(f"🎬 Фінальне відео збережено: {result}")
