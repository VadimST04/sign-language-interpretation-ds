import os
from audio_extractor import AudioExtractor
from speech_recognizer import SpeechRecognizer
from translator import Translator
from subtitle_generator import SubtitleGenerator
from video_subtitler import VideoSubtitler


class SubtitlePipeline:
    """
    Full pipeline:
        1. Extract audio
        2. Transcribe speech
        3. Translate
        4. Generate SRT
        5. Burn subtitles
    """

    def __init__(self, whisper_model: str = "medium", translator_model: str = "Helsinki-NLP/opus-mt-en-uk"):
        self.audio_extractor = AudioExtractor()
        self.speech_recognizer = SpeechRecognizer(whisper_model)
        self.translator = Translator(translator_model)
        self.subtitle_generator = SubtitleGenerator()
        self.video_subtitler = VideoSubtitler()

    def process(self, video_path: str, output_path: str = "video_with_subtitles.mp4") -> str:

        print("=" * 70)
        print("🚀 ЗАПУСК PIPELINE")
        print("=" * 70)

        # 1. Extract audio
        audio_path = self.audio_extractor.extract(video_path)

        # 2. Transcribe
        transcription = self.speech_recognizer.transcribe(audio_path)

        # 3. Translate
        translated_segments = self.translator.translate_segments(transcription["segments"])

        # 4. Create SRT
        srt_path = self.subtitle_generator.generate_srt(translated_segments, "temp_subtitles.srt")

        # 5. Burn subs
        self.video_subtitler.burn_subtitles(video_path, srt_path, output_path)

        # Cleanup
        try:
            os.remove(audio_path)
            os.remove(srt_path)
        except OSError:
            pass

        print("=" * 70)
        print(f"✅ ГОТОВО! Відео з субтитрами: {output_path}")
        print("=" * 70)
        return output_path
