from typing import List, Dict


class SubtitleGenerator:

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds to SRT timestamp."""
        ms = int((seconds % 1) * 1000)
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

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
