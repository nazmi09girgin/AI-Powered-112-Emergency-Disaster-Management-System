"""
AKOM Yardımcı Fonksiyonlar
"""

import pandas as pd
import os

# İstanbul ilçe koordinatları
ILCE_KOORDINATLARI = {
    "Avcılar": {"lat": 40.9792, "lon": 28.7214},
    "Kadıköy": {"lat": 40.9927, "lon": 29.0277},
    "Beşiktaş": {"lat": 41.0430, "lon": 29.0094},
    "Beyoğlu": {"lat": 41.0370, "lon": 28.9770},
    "Fatih": {"lat": 41.0186, "lon": 28.9397},
    "Şişli": {"lat": 41.0602, "lon": 28.9877},
    "Üsküdar": {"lat": 41.0234, "lon": 29.0152},
    "Bakırköy": {"lat": 40.9819, "lon": 28.8772},
    "Sarıyer": {"lat": 41.1667, "lon": 29.0500},
    "Maltepe": {"lat": 40.9346, "lon": 29.1296},
    "Kartal": {"lat": 40.8903, "lon": 29.1856},
    "Pendik": {"lat": 40.8761, "lon": 29.2336},
    "Bağcılar": {"lat": 41.0364, "lon": 28.8567},
    "Bahçelievler": {"lat": 41.0019, "lon": 28.8614},
    "Esenyurt": {"lat": 41.0333, "lon": 28.6833},
    "Beylikdüzü": {"lat": 41.0000, "lon": 28.6333},
    "Büyükçekmece": {"lat": 41.0167, "lon": 28.5833},
    "Silivri": {"lat": 41.0736, "lon": 28.2469},
    "Çatalca": {"lat": 41.1439, "lon": 28.4614},
    "Arnavutköy": {"lat": 41.1833, "lon": 28.7333},
    "Başakşehir": {"lat": 41.0939, "lon": 28.8011},
    "Esenler": {"lat": 41.0436, "lon": 28.8756},
    "Gaziosmanpaşa": {"lat": 41.0667, "lon": 28.9167},
    "Eyüpsultan": {"lat": 41.0500, "lon": 28.9333},
    "Kağıthane": {"lat": 41.0833, "lon": 28.9667},
    "Sultangazi": {"lat": 41.1000, "lon": 28.8667},
    "Ataşehir": {"lat": 40.9833, "lon": 29.1167},
    "Ümraniye": {"lat": 41.0167, "lon": 29.1167},
    "Sancaktepe": {"lat": 41.0000, "lon": 29.2333},
    "Sultanbeyli": {"lat": 40.9667, "lon": 29.2667},
    "Çekmeköy": {"lat": 41.0333, "lon": 29.1667},
    "Beykoz": {"lat": 41.1333, "lon": 29.1000},
    "Şile": {"lat": 41.1756, "lon": 29.6128},
    "Adalar": {"lat": 40.8761, "lon": 29.0906},
    "Tuzla": {"lat": 40.8167, "lon": 29.3000},
}


def get_ilce_koordinat(ilce_adi):
    """İlçe adından koordinat döndür"""
    if ilce_adi and ilce_adi in ILCE_KOORDINATLARI:
        return ILCE_KOORDINATLARI[ilce_adi]
    return {"lat": 41.0082, "lon": 28.9784}  # İstanbul merkez


def load_dataset():
    """Veri setini yükle"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), "data", "akom_dataset.csv")
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def save_analysis(analysis_result, ihbar_text):
    """Yeni analizi CSV'ye kaydet"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(script_dir), "data", "akom_dataset.csv")
    
    # Koordinatları al
    ilce = analysis_result.get('ilce')
    coords = get_ilce_koordinat(ilce)
    
    # Yeni satır oluştur
    new_row = {
        'ihbar_metni': ihbar_text,
        'olay_turu': analysis_result['olay_turu'],
        'oncelik': analysis_result['oncelik'],
        'birim': analysis_result['birim'],
        'ilce': ilce or 'İstanbul',
        'lat': coords['lat'],
        'lon': coords['lon']
    }
    
    # CSV'ye ekle
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    
    df.to_csv(data_path, index=False)
    return True


def get_priority_color(oncelik):
    """Öncelik seviyesine göre renk döndür"""
    colors = {
        "Kritik": "#FF0000",   # Kırmızı
        "Yüksek": "#FFA500",   # Turuncu
        "Orta": "#FFD700"      # Sarı
    }
    return colors.get(oncelik, "#808080")


def get_event_icon(olay_turu):
    """Olay türüne göre ikon döndür"""
    icons = {
        "Deprem": "🏚️",
        "Sel Baskını": "🌊",
        "Orman Yangını": "🔥",
        "Kar Fırtınası": "❄️",
        "Heyelan": "⛰️",
        "Trafik Kazası": "🚗",
        "Metro/Tünel Kazası": "🚇",
        "Bina Yangını": "🔥",
        "Gaz Kaçağı": "💨",
        "Altyapı Arızası": "🔧"
    }
    return icons.get(olay_turu, "📍")
