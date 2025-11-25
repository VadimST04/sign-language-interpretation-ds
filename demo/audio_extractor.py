import subprocess


class AudioExtractor:
    """Extract audio from video using FFmpeg."""

    def extract(self, video_path: str) -> str:
        audio_path = "temp_audio.wav"
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-ac", "1", "-ar", "16000",
            "-y", audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"🎧 Аудіо витягнуто: {audio_path}")
        return audio_path
