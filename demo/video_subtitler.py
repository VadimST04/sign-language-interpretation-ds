import subprocess


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
