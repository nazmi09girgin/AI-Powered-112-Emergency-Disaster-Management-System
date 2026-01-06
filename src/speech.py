"""
AKOM Ses İşleme Modülü
Whisper ile sesli ihbar girişi
"""

import whisper
import tempfile
import os
import subprocess

# Whisper modelini global olarak cache'le
_whisper_model = None


def check_ffmpeg():
    """FFmpeg'in yüklü ve erişilebilir olup olmadığını kontrol et"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            return True
    except FileNotFoundError:
        pass
    
    # Windows'ta yaygın FFmpeg konumlarını kontrol et
    common_paths = [
        r"C:\ffmpeg\bin",
        r"C:\Program Files\ffmpeg\bin",
        os.path.expanduser(r"~\AppData\Local\Microsoft\WinGet\Packages"),
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            ffmpeg_exe = os.path.join(path, "ffmpeg.exe")
            if os.path.exists(ffmpeg_exe):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
                return True
            if "WinGet" in path:
                for folder in os.listdir(path) if os.path.exists(path) else []:
                    if "FFmpeg" in folder:
                        ffmpeg_dir = os.path.join(path, folder)
                        for root, dirs, files in os.walk(ffmpeg_dir):
                            if "ffmpeg.exe" in files:
                                os.environ["PATH"] = root + os.pathsep + os.environ.get("PATH", "")
                                return True
    return False


def get_whisper_model():
    """Whisper modelini yükle ve cache'le"""
    global _whisper_model
    if _whisper_model is None:
        check_ffmpeg()
        
        # "large" model - en iyi Türkçe tanıma
        # İlk seferde ~3GB indirilecek, sonra cache'ten yüklenecek
        print("Whisper 'large' modeli yükleniyor (en iyi Türkçe desteği)...")
        _whisper_model = whisper.load_model("large")
        print("Model yüklendi!")
    return _whisper_model


def transcribe_audio(audio_bytes):
    """
    Ses dosyasını metne çevir
    
    Args:
        audio_bytes: Ses dosyasının byte içeriği
    
    Returns:
        str: Çevrilen metin
    """
    tmp_path = None
    try:
        check_ffmpeg()
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        print(f"Ses dosyası: {tmp_path}, Boyut: {os.path.getsize(tmp_path)} bytes")
        
        # Türkçe afet ihbarları için bağlam
        initial_prompt = """
        Bu bir afet ihbarıdır. İstanbul'dan acil durum bildirimi.
        Deprem, sel baskını, yangın, trafik kazası, gaz kaçağı, heyelan.
        Avcılar, Kadıköy, Beşiktaş, Esenyurt, Sarıyer, Bakırköy gibi ilçeler.
        Mahalle, cadde, sokak isimleri içerebilir.
        """
        
        model = get_whisper_model()
        result = model.transcribe(
            tmp_path,
            language="tr",
            fp16=True,  # GPU için aktif - çok daha hızlı
            initial_prompt=initial_prompt,
            beam_size=5,
            best_of=5,
            temperature=0
        )
        
        text = result["text"].strip()
        print(f"Sonuç: {text}")
        
        return text if text else None
    
    except Exception as e:
        import traceback
        print(f"Hata: {e}")
        print(traceback.format_exc())
        return None
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass


def is_whisper_available():
    """Whisper'ın kullanılabilir olup olmadığını kontrol et"""
    try:
        import whisper
        return True
    except ImportError:
        return False
