"""
AKOM Model - Turkish BERT ile İhbar Sınıflandırma
"""

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

# Olay türleri, öncelikler ve birimler
OLAY_TURLERI = ["Deprem", "Sel Baskını", "Orman Yangını", "Kar Fırtınası", 
                "Heyelan", "Trafik Kazası", "Metro/Tünel Kazası", 
                "Bina Yangını", "Gaz Kaçağı", "Altyapı Arızası",
                "Patlama", "Kimyasal Kaza"]

ONCELIKLER = ["Kritik", "Yüksek", "Orta"]

BIRIMLER = ["AFAD", "İtfaiye", "İSKİ", "İGDAŞ", "Yol Bakım ve Altyapı", 
            "Sağlık Ekipleri", "Kurtarma Ekipleri", "Trafik Ekipleri",
            "Metro İstanbul", "Orman Bölge Müdürlüğü", "BEDAŞ", "Ulaşım Daire Başkanlığı"]

# Olay türü anahtar kelimeleri
OLAY_ANAHTAR_KELIMELER = {
    "Deprem": [
        "deprem", "sarsıntı", "sismik", "enkaz", "çöktü", "yıkıldı", "hasar", "göçtü",
        "sallandı", "sallanıyor", "yer sallandı", "bina sallandı", "artçı", "artçı sarsıntı",
        "deprem oldu", "deprem hissedildi", "7 büyüklüğünde", "6 büyüklüğünde", "5 büyüklüğünde",
        "richter", "büyüklüğünde deprem", "şiddetinde deprem", "merkez üssü", "fay hattı",
        "bina çöktü", "duvarlar çatladı", "çatlak oluştu", "yıkılma tehlikesi"
    ],
    "Sel Baskını": ["sel", "su baskını", "taşkın", "dere taştı", "su altında", "yağmur", "kanalizasyon taştı", "su bastı"],
    "Orman Yangını": ["orman yangını", "ormanlık alanda", "ormanlık", "çalılık yangın", "duman yükseliyor", "orman", "alevler yayılıyor", "yangın çıktı"],
    "Kar Fırtınası": ["kar", "tipi", "buzlanma", "kar yağışı", "karla mücadele", "yollar kapandı"],
    "Heyelan": ["heyelan", "toprak kayması", "kaya düşmesi", "yamaç"],
    "Trafik Kazası": ["trafik kazası", "araç kazası", "zincirleme kaza", "araç devrildi", "otobüs kazası", "kaza oldu", "kaza", "çarpışma"],
    "Metro/Tünel Kazası": ["metro", "tünel", "tramvay", "raylı sistem"],
    "Bina Yangını": ["apartman yangın", "fabrika yangın", "iş yeri yangın", "çatı yangın", "daire yangın", "bina yangın"],
    "Gaz Kaçağı": ["gaz kaçağı", "gaz kokusu", "doğalgaz", "gaz borusu"],
    "Altyapı Arızası": ["su borusu patladı", "elektrik kesintisi", "trafo arıza", "yol çöktü", "su kesintisi", "kanalizasyon tıkandı"],
    "Patlama": ["patlama", "patladı", "infilak", "bomba", "patlayıcı", "şiddetli patlama", "patlama sesi"],
    "Kimyasal Kaza": ["kimyasal", "zehirli", "tehlikeli madde", "kimyasal koku", "kimyasal sızıntı", "asit", "radyasyon", "biyolojik"]
}

# Birim eşleştirmeleri
OLAY_BIRIM_ESLESME = {
    "Deprem": ["AFAD", "Kurtarma Ekipleri", "Sağlık Ekipleri"],
    "Sel Baskını": ["İSKİ", "İtfaiye", "AFAD"],
    "Orman Yangını": ["İtfaiye", "Orman Bölge Müdürlüğü", "AFAD"],
    "Kar Fırtınası": ["Yol Bakım ve Altyapı", "AFAD", "Ulaşım Daire Başkanlığı"],
    "Heyelan": ["AFAD", "Yol Bakım ve Altyapı", "Kurtarma Ekipleri"],
    "Trafik Kazası": ["Trafik Ekipleri", "Sağlık Ekipleri", "İtfaiye"],
    "Metro/Tünel Kazası": ["Metro İstanbul", "İtfaiye", "Sağlık Ekipleri"],
    "Bina Yangını": ["İtfaiye", "Sağlık Ekipleri", "İGDAŞ"],
    "Gaz Kaçağı": ["İGDAŞ", "İtfaiye", "AFAD"],
    "Altyapı Arızası": ["İSKİ", "İGDAŞ", "BEDAŞ", "Yol Bakım ve Altyapı"],
    "Patlama": ["AFAD", "İtfaiye", "Sağlık Ekipleri", "Kurtarma Ekipleri"],
    "Kimyasal Kaza": ["AFAD", "İtfaiye", "Sağlık Ekipleri"]
}

# Öncelik anahtar kelimeleri
ONCELIK_ANAHTAR_KELIMELER = {
    "Kritik": [
        "acil", "kritik", "mahsur kaldı", "enkaz", "çöktü", "patlama", "yaralı", "ölüm", 
        "can kaybı", "tahliye", "acilen", "şiddetli", "büyük", "yoğun", "hızla", "kuvvetli", 
        "şiddet", "yıkıldı", "göçtü", "hayati", "ölü", "ağır yaralı", "mahsur", 
        "7 büyüklüğünde", "6 büyüklüğünde", "kurtarma", "enkaz altında", "binalar yıkıldı"
    ],
    "Yüksek": [
        "tehlike", "risk", "büyüyor", "yayılıyor", "hasar", "zarar", "mağdur", "ciddi", 
        "ağır", "çatlak", "tehlikeli", "5 büyüklüğünde", "hafif yaralı", "yıkılma riski",
        "tahliye edilmeli", "bina boşaltılıyor"
    ],
    "Orta": ["arıza", "kesinti", "tıkandı", "şüpheli", "kontrol", "tespit", "hafif", "küçük"]
}


class AKOMClassifier:
    """AKOM İhbar Sınıflandırıcı"""
    
    def __init__(self, use_bert=True):
        self.use_bert = use_bert
        self.tokenizer = None
        self.model = None
        
        if use_bert:
            self._load_bert_model()
    
    def _load_bert_model(self):
        """Turkish BERT modelini yükle"""
        try:
            print("Turkish BERT modeli yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
            self.model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased")
            self.model.eval()
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"BERT modeli yüklenemedi: {e}")
            print("Kural tabanlı sınıflandırma kullanılacak.")
            self.use_bert = False
    
    def get_text_embedding(self, text):
        """Metin için BERT embedding al"""
        if not self.use_bert or self.model is None:
            return None
        
        inputs = self.tokenizer(text, return_tensors="pt", 
                               max_length=512, truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # CLS token embedding'i kullan
            embedding = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embedding
    
    def classify_event_type(self, text):
        """Olay türünü sınıflandır"""
        text_lower = text.lower()
        
        scores = {}
        for olay, keywords in OLAY_ANAHTAR_KELIMELER.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[olay] = score
        
        if scores:
            return max(scores, key=scores.get)
        return "Diğer"
    
    def classify_priority(self, text):
        """Öncelik seviyesini belirle"""
        text_lower = text.lower()
        
        # Kritik anahtar kelimeler
        for kw in ONCELIK_ANAHTAR_KELIMELER["Kritik"]:
            if kw in text_lower:
                return "Kritik"
        
        # Yüksek anahtar kelimeler
        for kw in ONCELIK_ANAHTAR_KELIMELER["Yüksek"]:
            if kw in text_lower:
                return "Yüksek"
        
        # Orta anahtar kelimeler veya varsayılan
        return "Orta"
    
    def assign_units(self, olay_turu):
        """İlgili birimleri ata - tümünü döndür"""
        if olay_turu in OLAY_BIRIM_ESLESME:
            return OLAY_BIRIM_ESLESME[olay_turu]  # Tüm birimleri döndür
        return ["AFAD"]
    
    def extract_location(self, text):
        """Metinden konum bilgisi çıkar"""
        # OSB ve önemli bölge eşleştirmeleri (bölge adı -> ilçe)
        onemli_bolgeler = {
            "Tuzla OSB": "Tuzla",
            "Tuzla Organize Sanayi": "Tuzla",
            "Dudullu OSB": "Ümraniye",
            "Dudullu Organize Sanayi": "Ümraniye",
            "İkitelli OSB": "Başakşehir",
            "İkitelli Organize Sanayi": "Başakşehir",
            "Beylikdüzü OSB": "Beylikdüzü",
            "Beylikdüzü Organize Sanayi": "Beylikdüzü",
            "Esenyurt OSB": "Esenyurt",
            "Anadolu Yakası OSB": "Tuzla",
            "Avrupa Yakası OSB": "Başakşehir",
            "Organize Sanayi Bölgesi": None,  # Genel OSB (ilçe belirsiz)
            "Sanayi Bölgesi": None,
            "İstanbul Havalimanı": "Arnavutköy",
            "Sabiha Gökçen": "Pendik",
            "Atatürk Havalimanı": "Bakırköy",
            "Kadıköy İskelesi": "Kadıköy",
            "Eminönü İskelesi": "Fatih",
            "Haydarpaşa Limanı": "Kadıköy",
            "Ambarlı Limanı": "Avcılar",
        }
        
        # Önce OSB/bölge ara
        found_bolge = None
        found_ilce = None
        
        for bolge, ilce in onemli_bolgeler.items():
            if bolge.lower() in text.lower():
                found_bolge = bolge
                if ilce:
                    found_ilce = ilce
                break
        
        # İlçe bulunamadıysa normal ilçe araması yap
        if not found_ilce:
            ilceler = [
                "Avcılar", "Kadıköy", "Beşiktaş", "Beyoğlu", "Fatih", "Şişli",
                "Üsküdar", "Bakırköy", "Sarıyer", "Maltepe", "Kartal", "Pendik",
                "Bağcılar", "Bahçelievler", "Esenyurt", "Beylikdüzü", "Büyükçekmece",
                "Silivri", "Çatalca", "Arnavutköy", "Başakşehir", "Esenler",
                "Gaziosmanpaşa", "Eyüpsultan", "Kağıthane", "Sultangazi", "Ataşehir",
                "Ümraniye", "Sancaktepe", "Sultanbeyli", "Çekmeköy", "Beykoz",
                "Şile", "Adalar", "Tuzla"
            ]
            
            for ilce in ilceler:
                if ilce in text:
                    found_ilce = ilce
                    break
        
        # Mahalle deseni
        mahalle_pattern = r'(\w+)\s+Mahallesi'
        mahalle_match = re.search(mahalle_pattern, text)
        found_mahalle = mahalle_match.group(1) if mahalle_match else None
        
        # Cadde deseni (Caddesi, caddesinde, Caddesi'nde, caddede vb.)
        cadde_pattern = r'([A-ZÇĞİÖŞÜa-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜa-zçğıöşü]+)*)\s+[Cc]adde(?:si|sinde|de)'
        cadde_match = re.search(cadde_pattern, text)
        found_cadde = cadde_match.group(1) if cadde_match else None
        
        # Sokak deseni (Sokak, Sokağı, sokağında vb.)
        sokak_pattern = r'(\w+(?:\s+\w+)?)\s+[Ss]oka(?:k|ğı|ğında)'
        sokak_match = re.search(sokak_pattern, text)
        found_sokak = sokak_match.group(1) if sokak_match else None
        
        # Alternatif sokak deseni
        if not found_sokak:
            sokak_pattern2 = r'(\w+(?:\s+\w+)?)\s+Sokağı'
            sokak_match2 = re.search(sokak_pattern2, text)
            found_sokak = sokak_match2.group(1) if sokak_match2 else None
        
        return {
            "ilce": found_ilce,
            "mahalle": found_mahalle,
            "cadde": found_cadde,
            "sokak": found_sokak,
            "bolge": found_bolge
        }
    
    def analyze(self, ihbar_text):
        """İhbar metnini analiz et ve tüm sınıflandırmaları döndür"""
        olay_turu = self.classify_event_type(ihbar_text)
        oncelik = self.classify_priority(ihbar_text)
        birimler = self.assign_units(olay_turu)
        konum = self.extract_location(ihbar_text)
        
        # BERT embedding (opsiyonel kullanım için)
        embedding = None
        if self.use_bert:
            embedding = self.get_text_embedding(ihbar_text)
        
        return {
            "olay_turu": olay_turu,
            "oncelik": oncelik,
            "birim": birimler[0],  # İlk birim (ana birim)
            "birimler": birimler,  # Tüm birimler
            "ilce": konum["ilce"],
            "mahalle": konum["mahalle"],
            "cadde": konum["cadde"],
            "sokak": konum["sokak"],
            "bolge": konum["bolge"],
            "embedding": embedding
        }


def test_model():
    """Model testleri"""
    classifier = AKOMClassifier(use_bert=False)  # Hızlı test için BERT olmadan
    
    test_ihbarlar = [
        "Avcılar Cihangir Mahallesi Cumhuriyet Caddesi yakınlarında şiddetli deprem hissedildi, binalarda hasar var.",
        "Kadıköy Fenerbahçe Mahallesi'nde yoğun yağış nedeniyle sel baskını yaşanıyor. Alt geçitler su altında.",
        "Sarıyer ormanlık alanda büyük yangın çıktı. Alevler rüzgarla yayılıyor.",
        "Esenyurt'ta zincirleme trafik kazası meydana geldi. Çok sayıda yaralı var.",
        "Beşiktaş Levent'te gaz kaçağı var, koku çok yoğun. Patlama riski mevcut."
    ]
    
    print("\n=== Model Test Sonuçları ===\n")
    for ihbar in test_ihbarlar:
        sonuc = classifier.analyze(ihbar)
        print(f"İhbar: {ihbar[:80]}...")
        print(f"  Olay Türü: {sonuc['olay_turu']}")
        print(f"  Öncelik: {sonuc['oncelik']}")
        print(f"  Birim: {sonuc['birim']}")
        print(f"  İlçe: {sonuc['ilce']}")
        print(f"  Mahalle: {sonuc['mahalle']}")
        print()


if __name__ == "__main__":
    test_model()
