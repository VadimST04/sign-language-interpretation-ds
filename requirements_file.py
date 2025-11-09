import sys

def test_imports():
    """Тестирование импортов основных библиотек"""
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ УСТАНОВКИ")
    print("=" * 60)
    
    tests = [
        ("PyTorch", "torch"),
        ("Whisper", "whisper"),
        ("OpenCV", "cv2"),
        ("NumPy", "numpy"),
        ("Transformers", "transformers"),
        ("FFmpeg-python", "ffmpeg"),
        ("MediaPipe", "mediapipe"),
        ("spaCy", "spacy"),
        ("FastAPI", "fastapi"),
        ("Editdistance", "editdistance"),
    ]
    
    passed = 0
    failed = 0
    
    for name, module in tests:
        try:
            __import__(module)
            print(f"✓ {name:20s} OK")
            passed += 1
        except ImportError as e:
            print(f"✗ {name:20s} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Результат: {passed} успешно, {failed} неудачно")
    print("=" * 60)
    
    if failed > 0:
        print("\nУстановите недостающие пакеты:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✓ Все зависимости установлены корректно!")

def test_gpu():
    """Проверка доступности GPU"""
    import torch
    
    print("\n" + "=" * 60)
    print("ПРОВЕРКА GPU")
    print("=" * 60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA доступна")
        print(f"  Устройство: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("! GPU не обнаружен. Будет использоваться CPU")
        print("  Для GPU поддержки установите CUDA и соответствующую версию PyTorch")
    
    print("=" * 60)

def test_ffmpeg():
    """Проверка FFmpeg"""
    import subprocess
    
    print("\n" + "=" * 60)
    print("ПРОВЕРКА FFMPEG")
    print("=" * 60)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               capture_output=True, text=True)
        version = result.stdout.split('\n')[0]
        print(f"✓ FFmpeg установлен: {version}")
    except FileNotFoundError:
        print("✗ FFmpeg не найден!")
        print("  Установите FFmpeg для обработки видео")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: https://ffmpeg.org/download.html")
    
    print("=" * 60)

if __name__ == "__main__":
    test_imports()
    test_gpu()
    test_ffmpeg()
    
    print("\n✓ Система готова к работе!")
    print("\nДля запуска pipeline:")
    print("  python sign_language_pipeline.py")
