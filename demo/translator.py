from transformers import MarianMTModel, MarianTokenizer


class Translator:

    def __init__(self, model_name: str):
        print(f"Завантаження моделі перекладу: {model_name}")
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

    def translate_segments(self, segments):
        new_segments = []
        for seg in segments:
            text = seg["text"]
            batch = self.tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")
            translated = self.model.generate(**batch)
            out = self.tokenizer.decode(translated[0], skip_special_tokens=True)

            new_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": out
            })

        print("🌍 Переклад завершено")
        return new_segments
