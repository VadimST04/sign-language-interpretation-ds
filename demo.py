"""
English Subtitles Pipeline (Whisper -> SRT -> Burned-in)

This module extracts audio from a video, transcribes English speech with Whisper,
generates SRT subtitles from the transcription segments, and burns the subtitles
into the original video using FFmpeg.
"""

import os
import subprocess
from typing import Dict, List
import torch
import whisper


class AudioExtractor:
    """Extracts an audio track from a video file using FFmpeg."""

    def extract(self, video_path: str, output_audio: str = "temp_audio.wav") -> str:
        """
        Extract audio from the given video file (16 kHz, mono WAV).

        Args:
            video_path: Path to the input video file.
            output_audio: Output path to save the extracted audio.

        Returns:
            Path to the extracted audio file.
        """
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
    """Performs English speech recognition using OpenAI Whisper."""

    def __init__(self, model_size: str = "medium", device: str = None):
        """
        Initialize the Whisper model.

        Args:
            model_size: Whisper model size ('tiny'/'base'/'small'/'medium'/'large').
            device: Device to run the model on ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Використовується пристрій для Whisper: {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe_english(self, audio_path: str) -> Dict:
        """
        Transcribe English audio to text using Whisper with segment timestamps.

        Args:
            audio_path: Path to the WAV audio file.

        Returns:
            Whisper transcription result dict (includes 'text' and 'segments').
        """
        result = self.model.transcribe(
            audio_path,
            language="en",
            word_timestamps=False  # segments are enough for stable subtitles
        )
        print(f"✓ Розпізнано англійський текст: {result['text'][:100]}...")
        return result


class SubtitleGenerator:
    """Creates SRT subtitle files from Whisper transcription segments."""

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """
        Convert float seconds to SRT timestamp (HH:MM:SS,mmm).

        Args:
            seconds: Time in seconds.

        Returns:
            Timestamp string in SRT format.
        """
        if seconds < 0:
            seconds = 0.0
        hrs = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        return f"{hrs:02}:{mins:02}:{secs:02},{ms:03}"

    def generate_srt(self, segments: List[Dict], output_path: str = "subtitles.srt") -> str:
        """
        Generate an SRT file from Whisper segments (English text).

        Args:
            segments: List of Whisper segments with 'start', 'end', and 'text'.
            output_path: Output SRT file path.

        Returns:
            Path to the generated SRT file.
        """
        # Small timing safety: ensure non-zero durations and trim whitespace
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, start=1):
                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", start + 0.01))
                if end <= start:
                    end = start + 0.50  # ensure visible duration

                start_ts = self._format_timestamp(start)
                end_ts = self._format_timestamp(end)
                text = str(seg.get("text", "")).strip().replace("\n", " ")

                f.write(f"{i}\n{start_ts} --> {end_ts}\n{text}\n\n")

        print(f"✓ Субтитри (SRT) створено: {output_path}")
        return output_path


class VideoSubtitler:
    """Burns SRT subtitles into a video using FFmpeg."""

    @staticmethod
    def _escape_path_for_ffmpeg(path: str) -> str:
        """
        Escape file path for FFmpeg subtitles filter (especially on Windows).

        Args:
            path: File path to the SRT file.

        Returns:
            Escaped path string safe for FFmpeg filter usage.
        """
        # FFmpeg filter expects backslashes and colons escaped.
        esc = path.replace("\\", "\\\\").replace(":", "\\:")
        return esc

    def burn_subtitles(self, input_video: str, srt_path: str, output_video: str) -> str:
        """
        Burn subtitles into the video using FFmpeg 'subtitles' filter.

        Args:
            input_video: Path to the source video.
            srt_path: Path to the SRT subtitle file.
            output_video: Path to the output video with hardcoded subtitles.

        Returns:
            Path to the output video file.
        """
        srt_escaped = self._escape_path_for_ffmpeg(srt_path)
        # You can tweak styling via ASS force_style
        vf = f"subtitles='{srt_escaped}':force_style='Fontsize=26,Outline=2,Shadow=1,MarginV=24'"

        cmd = [
            "ffmpeg", "-i", input_video,
            "-vf", vf,
            "-c:a", "copy",
            "-y", output_video
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Субтитри впалено у відео: {output_video}")
        return output_video


class EnglishSubtitlesPipeline:
    """
    End-to-end pipeline to add English burned-in subtitles to a video:

        1) Extract audio from video.
        2) Transcribe English speech with Whisper.
        3) Generate SRT subtitles from segments.
        4) Burn subtitles into the original video with FFmpeg.
    """

    def __init__(self, whisper_model: str = "medium"):
        """
        Initialize all components.

        Args:
            whisper_model: Whisper model size (e.g., 'small', 'medium', 'large').
        """
        self.audio_extractor = AudioExtractor()
        self.speech_recognizer = SpeechRecognizer(model_size=whisper_model)
        self.subtitle_generator = SubtitleGenerator()
        self.video_subtitler = VideoSubtitler()

    def process(self, video_path: str, output_path: str = "video_with_subs.mp4") -> str:
        """
        Run the full pipeline to produce a video with English subtitles.

        Args:
            video_path: Path to the input video.
            output_path: Path to save the subtitled video.

        Returns:
            Path to the final video with burned-in subtitles.
        """
        print("=" * 60)
        print("🚀 ЗАПУСК PIPELINE: АНГЛОМОВНІ СУБТИТРИ (Whisper → SRT → burn-in)")
        print("=" * 60)

        # 1. Extract audio
        audio_path = self.audio_extractor.extract(video_path)

        # 2. Transcribe English
        transcription = self.speech_recognizer.transcribe_english(audio_path)

        # 3. Build SRT from segments
        if "segments" not in transcription or not transcription["segments"]:
            raise RuntimeError("Не вдалося отримати сегменти для субтитрів з Whisper.")
        srt_path = self.subtitle_generator.generate_srt(transcription["segments"], "temp_subtitles.srt")

        # 4. Burn subtitles into original video
        self.video_subtitler.burn_subtitles(video_path, srt_path, output_path)

        # Cleanup
        try:
            os.remove(audio_path)
            os.remove(srt_path)
        except OSError:
            pass

        print("=" * 60)
        print(f"✅ ГОТОВО! Вихідний файл: {output_path}")
        print("=" * 60)
        return output_path


if __name__ == "__main__":
    pipeline = EnglishSubtitlesPipeline(whisper_model="medium")

    input_video = r"D:\PyCharm\PyProjects\IntPreparation\ds-interpretation\input_video.mp4"
    output_video = "output_with_english_subtitles.mp4"

    result = pipeline.process(video_path=input_video, output_path=output_video)
    print(f"🎬 Фінальне відео збережено: {result}")
