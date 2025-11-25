import subprocess


class VideoSubtitler:
    """Burns subtitles into a video using FFmpeg."""

    @staticmethod
    def _escape_path(path: str) -> str:
        return path.replace("\\", "\\\\").replace(":", "\\:")

    @staticmethod
    def _hex_to_ass(hex_color: str) -> str:
        """
        Convert HEX (#RRGGBB) → ASS color format (&H00BBGGRR)
        Required by FFmpeg.
        """

        if not isinstance(hex_color, str):
            raise ValueError("Color must be HEX string '#RRGGBB'")

        if not hex_color.startswith("#") or len(hex_color) != 7:
            raise ValueError(f"Invalid HEX color '{hex_color}'. Expected format '#RRGGBB'.")

        r = hex_color[1:3]
        g = hex_color[3:5]
        b = hex_color[5:7]

        # ASS format -> &HAABBGGRR
        return f"&H00{b}{g}{r}"

    def burn_subtitles(
        self,
        input_video: str,
        srt_path: str,
        output_video: str,
        font_size: int = 28,
        font_color: str = "#FFFFFF",
        font_family: str = "Arial"
    ) -> str:

        ass_color = self._hex_to_ass(font_color)

        style = (
            f"FontName={font_family},"
            f"FontSize={font_size},"
            f"PrimaryColour={ass_color},"
            f"Outline=2,Shadow=1,MarginV=25"
        )

        srt_escaped = self._escape_path(srt_path)
        vf = f"subtitles='{srt_escaped}':force_style='{style}'"

        cmd = [
            "ffmpeg", "-i", input_video,
            "-vf", vf,
            "-c:a", "copy",
            "-y", output_video
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        return output_video
