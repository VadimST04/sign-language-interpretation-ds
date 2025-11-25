import whisper


class SpeechRecognizer:
    def __init__(self, model_name="medium"):
        print(f"Завантаження Whisper моделі: {model_name}")
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_path: str):
        print("🔍 Транскрипція аудіо...")
        result = self.model.transcribe(audio_path)
        return result
