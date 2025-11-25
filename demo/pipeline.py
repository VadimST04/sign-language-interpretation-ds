import os
from audio_extractor import AudioExtractor
from speech_recognizer import SpeechRecognizer
from translator import Translator
from subtitle_generator import SubtitleGenerator
from video_subtitler import VideoSubtitler


class SubtitlePipeline:

    def __init__(self, whisper_model="medium", translator_model="Helsinki-NLP/opus-mt-en-uk"):
        self.audio_extractor = AudioExtractor()
        self.speech_recognizer = SpeechRecognizer(whisper_model)
        self.translator = Translator(translator_model)
        self.subtitle_generator = SubtitleGenerator()
        self.video_subtitler = VideoSubtitler()

    def process(
        self,
        video_path: str,
        output_path: str,
        font_size: int,
        font_color: str,
        font_family: str
    ):
        audio_path = self.audio_extractor.extract(video_path)
        transcription = self.speech_recognizer.transcribe(audio_path)
        translated_segments = self.translator.translate_segments(transcription["segments"])

        srt_path = self.subtitle_generator.generate_srt(
            translated_segments,
            "temp_subtitles.srt"
        )

        self.video_subtitler.burn_subtitles(
            video_path,
            srt_path,
            output_path,
            font_size=font_size,
            font_color=font_color,
            font_family=font_family
        )

        try:
            os.remove(audio_path)
            os.remove(srt_path)
        except OSError:
            pass

        return output_path
