"""
AKOM İhbar Veri Seti Oluşturucu
5000 satırlık sentetik veri seti üretir
"""

import pandas as pd
import numpy as np
import random
import os

# Rastgelelik için seed
random.seed(42)
np.random.seed(42)

# İstanbul ilçeleri ve koordinatları
ILCELER = {
    "Avcılar": {"lat": 40.9792, "lon": 28.7214, "mahalleler": ["Cihangir", "Denizköşkler", "Firuzköy", "Gümüşpala", "Mustafa Kemal Paşa", "Yeşilkent", "Ambarlı", "Tahtakale"]},
    "Kadıköy": {"lat": 40.9927, "lon": 29.0277, "mahalleler": ["Caferağa", "Fenerbahçe", "Göztepe", "Koşuyolu", "Moda", "Suadiye", "Bostancı", "Erenköy", "Fikirtepe"]},
    "Beşiktaş": {"lat": 41.0430, "lon": 29.0094, "mahalleler": ["Akatlar", "Bebek", "Etiler", "Levent", "Ortaköy", "Arnavutköy", "Dikilitaş", "Yıldız"]},
    "Beyoğlu": {"lat": 41.0370, "lon": 28.9770, "mahalleler": ["Cihangir", "Galata", "Karaköy", "Taksim", "Tarlabaşı", "Kasımpaşa", "Dolapdere"]},
    "Fatih": {"lat": 41.0186, "lon": 28.9397, "mahalleler": ["Aksaray", "Balat", "Eminönü", "Sultanahmet", "Vefa", "Karagümrük", "Saraçhane", "Zeyrek"]},
    "Şişli": {"lat": 41.0602, "lon": 28.9877, "mahalleler": ["Mecidiyeköy", "Nişantaşı", "Osmanbey", "Fulya", "Bomonti", "Halaskargazi", "Teşvikiye"]},
    "Üsküdar": {"lat": 41.0234, "lon": 29.0152, "mahalleler": ["Acıbadem", "Altunizade", "Bağlarbaşı", "Çengelköy", "Kuzguncuk", "Ümraniye", "Bulgurlu"]},
    "Bakırköy": {"lat": 40.9819, "lon": 28.8772, "mahalleler": ["Ataköy", "Bahçelievler", "Florya", "Yeşilköy", "Şenlikköy", "Osmaniye", "Kartaltepe"]},
    "Sarıyer": {"lat": 41.1667, "lon": 29.0500, "mahalleler": ["Baltalimanı", "Emirgan", "İstinye", "Maslak", "Tarabya", "Rumeli Feneri", "Yeniköy"]},
    "Maltepe": {"lat": 40.9346, "lon": 29.1296, "mahalleler": ["Altıntepe", "Bağlarbaşı", "Cevizli", "Girne", "Zümrütevler", "Küçükyalı", "İdealtepe"]},
    "Kartal": {"lat": 40.8903, "lon": 29.1856, "mahalleler": ["Cevizli", "Kordonboyu", "Soğanlık", "Uğur Mumcu", "Yakacık", "Hürriyet", "Esentepe"]},
    "Pendik": {"lat": 40.8761, "lon": 29.2336, "mahalleler": ["Batı", "Esenyalı", "Güzelyalı", "Kaynarca", "Kurtköy", "Yenişehir", "Velibaba"]},
    "Bağcılar": {"lat": 41.0364, "lon": 28.8567, "mahalleler": ["Barbaros", "Demirkapı", "Fevzi Çakmak", "Güneşli", "Kirazlı", "Mahmutbey", "Yıldıztepe"]},
    "Bahçelievler": {"lat": 41.0019, "lon": 28.8614, "mahalleler": ["Bahçelievler", "Kocasinan", "Soğanlı", "Şirinevler", "Yenibosna", "Zafer", "Cumhuriyet"]},
    "Esenyurt": {"lat": 41.0333, "lon": 28.6833, "mahalleler": ["Ardıçlı", "Fatih", "İnönü", "Kıraç", "Mehterçeşme", "Saadetdere", "Yenikent"]},
    "Beylikdüzü": {"lat": 41.0000, "lon": 28.6333, "mahalleler": ["Adnan Kahveci", "Barış", "Büyükşehir", "Cumhuriyet", "Dereağzı", "Gürpınar", "Yakuplu"]},
    "Büyükçekmece": {"lat": 41.0167, "lon": 28.5833, "mahalleler": ["Bahçelievler", "Fatih", "Güzelce", "Kumburgaz", "Mimarsinan", "Pınartepe", "Tepecik"]},
    "Silivri": {"lat": 41.0736, "lon": 28.2469, "mahalleler": ["Alibey", "Cumhuriyet", "Fatih", "Gümüşyaka", "Ortaköy", "Piri Mehmet Paşa", "Selimpaşa"]},
    "Çatalca": {"lat": 41.1439, "lon": 28.4614, "mahalleler": ["Ferhatpaşa", "Kaleiçi", "Kestanelik", "Subaşı", "Yalıköy", "Çiftlikköy"]},
    "Arnavutköy": {"lat": 41.1833, "lon": 28.7333, "mahalleler": ["Anadolu", "Boğazköy", "Haraççı", "İmrahor", "Taşoluk", "Yeşilbayır"]},
    "Başakşehir": {"lat": 41.0939, "lon": 28.8011, "mahalleler": ["Altınşehir", "Bahçeşehir 1. Kısım", "Bahçeşehir 2. Kısım", "Güvercintepe", "İkitelli", "Kayabaşı"]},
    "Esenler": {"lat": 41.0436, "lon": 28.8756, "mahalleler": ["Atışalanı", "Davutpaşa", "Fevzi Çakmak", "Kemer", "Menderes", "Oruçreis", "Tuna"]},
    "Gaziosmanpaşa": {"lat": 41.0667, "lon": 28.9167, "mahalleler": ["Bağlarbaşı", "Fevzi Çakmak", "Karadeniz", "Karlıtepe", "Mevlana", "Pazariçi", "Yıldıztabya"]},
    "Eyüpsultan": {"lat": 41.0500, "lon": 28.9333, "mahalleler": ["Alibeyköy", "Defterdar", "Güzeltepe", "İslambey", "Nişanca", "Rami", "Yeşilpınar"]},
    "Kağıthane": {"lat": 41.0833, "lon": 28.9667, "mahalleler": ["Çağlayan", "Gültepe", "Hamidiye", "Harmantepe", "Merkez", "Nurtepe", "Ortabayır"]},
    "Sultangazi": {"lat": 41.1000, "lon": 28.8667, "mahalleler": ["50. Yıl", "75. Yıl", "Cebeci", "Esentepe", "Gazi", "Sultançiftliği", "Yunusemre"]},
    "Ataşehir": {"lat": 40.9833, "lon": 29.1167, "mahalleler": ["Atatürk", "Barbaros", "Ferhatpaşa", "İçerenköy", "Küçükbakkalköy", "Yenisahra"]},
    "Ümraniye": {"lat": 41.0167, "lon": 29.1167, "mahalleler": ["Armağanevler", "Çakmak", "Ihlamurkuyu", "Kazım Karabekir", "Namık Kemal", "Tantavi"]},
    "Sancaktepe": {"lat": 41.0000, "lon": 29.2333, "mahalleler": ["Abdurrahman Gazi", "Emek", "Meclis", "Osmangazi", "Sarıgazi", "Yenidoğan"]},
    "Sultanbeyli": {"lat": 40.9667, "lon": 29.2667, "mahalleler": ["Abdurrahman Gazi", "Ahmet Yesevi", "Battalgazi", "Fatih", "Mecidiye", "Yavuz Selim"]},
    "Çekmeköy": {"lat": 41.0333, "lon": 29.1667, "mahalleler": ["Alemdağ", "Çatalmeşe", "Hamidiye", "Merkez", "Ömerli", "Taşdelen"]},
    "Beykoz": {"lat": 41.1333, "lon": 29.1000, "mahalleler": ["Acarlar", "Anadolu Hisarı", "Çubuklu", "Kavacık", "Paşabahçe", "Riva"]},
    "Şile": {"lat": 41.1756, "lon": 29.6128, "mahalleler": ["Ağva", "Balibey", "Hacılli", "Kumbaba", "Sahilköy", "Sofular"]},
    "Adalar": {"lat": 40.8761, "lon": 29.0906, "mahalleler": ["Burgazada", "Büyükada", "Heybeliada", "Kınalıada", "Sedefadası"]},
    "Tuzla": {"lat": 40.8167, "lon": 29.3000, "mahalleler": ["Aydınlı", "Aydıntepe", "Cami", "İçmeler", "Mimar Sinan", "Postane", "Şifa"]},
}

# Olay türleri ve ilişkili bilgiler
OLAY_TURLERI = {
    "Deprem": {
        "birimler": ["AFAD", "Kurtarma Ekipleri", "Sağlık Ekipleri", "İtfaiye"],
        "oncelik_dagilimi": {"Kritik": 0.6, "Yüksek": 0.3, "Orta": 0.1},
        "sablonlar": [
            "{konum}'de şiddetli deprem hissedildi, binalarda hasar var. İlgili ekipler acilen yönlendirilsin.",
            "{konum} bölgesinde deprem sonrası enkaz altında mahsur kalan vatandaşlar var. Acil müdahale gerekiyor.",
            "{konum}'de deprem nedeniyle bina çökmesi meydana geldi. Kurtarma ekipleri talep ediliyor.",
            "{konum} civarında deprem sonrası hasar tespit edildi, çatlaklar oluşmuş durumda.",
            "{konum}'de şiddetli sarsıntı hissedildi, panik yaşanıyor. Durum tespiti için ekip isteniyor.",
            "{konum} mahallesinde deprem sonrası gaz kaçağı şüphesi var. Acil müdahale lazım.",
            "{konum}'de deprem nedeniyle yol çökmesi meydana geldi. Trafik akışı durdu.",
        ]
    },
    "Sel Baskını": {
        "birimler": ["İSKİ", "İtfaiye", "AFAD", "Yol Bakım ve Altyapı"],
        "oncelik_dagilimi": {"Kritik": 0.4, "Yüksek": 0.4, "Orta": 0.2},
        "sablonlar": [
            "{konum}'de yoğun yağış nedeniyle sel baskını yaşanıyor. Alt geçitler su altında kaldı.",
            "{konum} bölgesinde dere taştı, evlere su girdi. Acil tahliye gerekiyor.",
            "{konum}'de aşırı yağmur sonrası sokaklar göle döndü. Araçlar mahsur kaldı.",
            "{konum} caddesinde kanalizasyon taştı. Vatandaşlar mağdur durumda.",
            "{konum}'de sel suları yükseliyor, bodrum katlar su altında. Acil pompa desteği isteniyor.",
            "{konum} mahallesinde su baskını nedeniyle elektrik kesintisi yaşanıyor.",
        ]
    },
    "Orman Yangını": {
        "birimler": ["İtfaiye", "Orman Bölge Müdürlüğü", "AFAD"],
        "oncelik_dagilimi": {"Kritik": 0.5, "Yüksek": 0.35, "Orta": 0.15},
        "sablonlar": [
            "{konum} yakınlarında orman yangını çıktı. Alevler hızla yayılıyor.",
            "{konum} bölgesinde ormanlık alanda büyük yangın var. Helikopter desteği gerekiyor.",
            "{konum}'de çalılık alanda yangın başladı. Rüzgar nedeniyle tehlike büyüyor.",
            "{konum} civarında orman yangını yerleşim yerlerine yaklaşıyor. Tahliye başlatılmalı.",
            "{konum}'de şüpheli duman yükseliyor. Yangın riski mevcut, kontrol edilmeli.",
        ]
    },
    "Kar Fırtınası": {
        "birimler": ["Yol Bakım ve Altyapı", "AFAD", "Ulaşım Daire Başkanlığı"],
        "oncelik_dagilimi": {"Kritik": 0.3, "Yüksek": 0.4, "Orta": 0.3},
        "sablonlar": [
            "{konum}'de yoğun kar yağışı nedeniyle yollar kapandı. Araçlar mahsur kaldı.",
            "{konum} bölgesinde buzlanma nedeniyle trafik kazaları yaşanıyor. Tuzlama ekibi isteniyor.",
            "{konum}'de kar kalınlığı artıyor, toplu taşıma durdu. Karla mücadele ekibi gerekli.",
            "{konum} mahallesinde yoğun kar nedeniyle çatılarda çökme riski var.",
            "{konum}'de tipi nedeniyle görüş mesafesi sıfıra düştü. Sürücüler uyarılmalı.",
            "{konum} civarında kar fırtınası sürüyor, elektrik direkleri devrildi.",
        ]
    },
    "Heyelan": {
        "birimler": ["AFAD", "Yol Bakım ve Altyapı", "Kurtarma Ekipleri"],
        "oncelik_dagilimi": {"Kritik": 0.5, "Yüksek": 0.35, "Orta": 0.15},
        "sablonlar": [
            "{konum}'de heyelan meydana geldi. Yol tamamen kapandı.",
            "{konum} bölgesinde toprak kayması nedeniyle evler tehlike altında.",
            "{konum}'de şiddetli yağış sonrası heyelan riski oluştu. Bölge boşaltılmalı.",
            "{konum} yakınlarında yamaçtan kaya düşmesi var. Yol trafiğe kapatıldı.",
            "{konum}'de heyelan sonrası araçlar toprak altında kaldı. Kurtarma ekibi şart.",
        ]
    },
    "Trafik Kazası": {
        "birimler": ["Trafik Ekipleri", "Sağlık Ekipleri", "İtfaiye", "Yol Bakım ve Altyapı"],
        "oncelik_dagilimi": {"Kritik": 0.4, "Yüksek": 0.4, "Orta": 0.2},
        "sablonlar": [
            "{konum}'de zincirleme trafik kazası meydana geldi. Çok sayıda yaralı var.",
            "{konum} kavşağında ağır tonajlı araç devrildi. Yol trafiğe kapandı.",
            "{konum}'de otobüs kazası yaşandı. Ambulans ve itfaiye acil gerekli.",
            "{konum} köprüsünde çoklu araç kazası var. Trafik felç oldu.",
            "{konum}'de trafik kazası sonrası yakıt sızıntısı var. Yangın riski mevcut.",
            "{konum} tünelinde kaza meydana geldi. Havalandırma sorunu yaşanıyor.",
        ]
    },
    "Metro/Tünel Kazası": {
        "birimler": ["Metro İstanbul", "İtfaiye", "Sağlık Ekipleri", "AFAD"],
        "oncelik_dagilimi": {"Kritik": 0.6, "Yüksek": 0.3, "Orta": 0.1},
        "sablonlar": [
            "{konum} metro istasyonunda teknik arıza nedeniyle yolcular mahsur kaldı.",
            "{konum}'de metro tünelinde yangın çıktı. Tahliye başlatıldı.",
            "{konum} metro hattında elektrik kesintisi var. Vagonlar tünelde durdu.",
            "{konum}'de tünel girişinde göçük meydana geldi. Trafik durdu.",
            "{konum} metro istasyonunda duman yayılıyor. Acil müdahale gerekli.",
        ]
    },
    "Bina Yangını": {
        "birimler": ["İtfaiye", "Sağlık Ekipleri", "İGDAŞ"],
        "oncelik_dagilimi": {"Kritik": 0.5, "Yüksek": 0.35, "Orta": 0.15},
        "sablonlar": [
            "{konum}'de apartmanda yangın çıktı. Üst katlarda mahsur kalanlar var.",
            "{konum} bölgesinde fabrikada büyük yangın var. Birden fazla itfaiye ekibi gerekli.",
            "{konum}'de iş yerinde yangın başladı. Dumandan etkilenenler var.",
            "{konum} apartmanında gaz patlaması sonrası yangın çıktı. Acil müdahale şart.",
            "{konum}'de çatı katında yangın var. Bitişik binalara sıçrama riski mevcut.",
            "{konum} sitesinde elektrik kontağından yangın çıktı. Tahliye ediliyor.",
        ]
    },
    "Gaz Kaçağı": {
        "birimler": ["İGDAŞ", "İtfaiye", "AFAD"],
        "oncelik_dagilimi": {"Kritik": 0.5, "Yüksek": 0.35, "Orta": 0.15},
        "sablonlar": [
            "{konum}'de yoğun gaz kokusu alınıyor. Patlama riski var, acil müdahale gerekli.",
            "{konum} sokağında doğalgaz borusu patladı. Bölge tahliye edilmeli.",
            "{konum}'de inşaat çalışması sırasında gaz hattı hasar gördü.",
            "{konum} apartmanında gaz kaçağı tespit edildi. Elektrik kesilmeli.",
            "{konum}'de ana gaz hattında sızıntı var. Trafik yönlendirilmeli.",
        ]
    },
    "Altyapı Arızası": {
        "birimler": ["İSKİ", "İGDAŞ", "BEDAŞ", "Yol Bakım ve Altyapı"],
        "oncelik_dagilimi": {"Kritik": 0.2, "Yüksek": 0.4, "Orta": 0.4},
        "sablonlar": [
            "{konum}'de ana su borusu patladı. Sokak su altında kaldı.",
            "{konum} bölgesinde elektrik trafosu arızalandı. Geniş çaplı kesinti var.",
            "{konum}'de yol çöktü, altyapı hasarı mevcut. Tehlike oluşturdu.",
            "{konum} mahallesinde uzun süredir su kesintisi yaşanıyor. Vatandaşlar şikayetçi.",
            "{konum}'de kanalizasyon tıkandı, kötü koku yayılıyor. Acil temizlik gerekli.",
            "{konum} caddesinde aydınlatma direkleri devrildi. Tehlike arz ediyor.",
        ]
    },
}

# Cadde ve sokak isimleri
CADDELER = [
    "Cumhuriyet Caddesi", "Atatürk Caddesi", "İstiklal Caddesi", "Bağdat Caddesi",
    "Halaskargazi Caddesi", "Barbaros Bulvarı", "Büyükdere Caddesi", "Vatan Caddesi",
    "Millet Caddesi", "Kennedy Caddesi", "Sahil Yolu", "Mecidiyeköy Yolu",
    "Beşiktaş Caddesi", "Kadıköy Caddesi", "Taksim Meydanı", "Kartal Caddesi"
]

SOKAKLAR = [
    "Yıldız Sokak", "Güneş Sokak", "Çiçek Sokak", "Gül Sokak", "Lale Sokak",
    "Papatya Sokak", "Menekşe Sokak", "Orkide Sokak", "Karanfil Sokak",
    "1. Sokak", "2. Sokak", "3. Sokak", "4. Sokak", "5. Sokak",
    "Atatürk Sokak", "Fatih Sokak", "Mimar Sinan Sokak", "Yunus Emre Sokak"
]


def konum_olustur():
    """Rastgele konum oluşturur"""
    ilce = random.choice(list(ILCELER.keys()))
    ilce_bilgi = ILCELER[ilce]
    mahalle = random.choice(ilce_bilgi["mahalleler"])
    cadde = random.choice(CADDELER)
    sokak = random.choice(SOKAKLAR)
    
    # Koordinatlara küçük rastgele sapma ekle
    lat = ilce_bilgi["lat"] + random.uniform(-0.02, 0.02)
    lon = ilce_bilgi["lon"] + random.uniform(-0.02, 0.02)
    
    konum_str = f"{ilce} {mahalle} Mahallesi {cadde} {sokak} yakınlarında"
    
    return {
        "konum_str": konum_str,
        "ilce": ilce,
        "mahalle": mahalle,
        "lat": lat,
        "lon": lon
    }


def oncelik_sec(dagilim):
    """Ağırlıklı dağılıma göre öncelik seçer"""
    secenekler = list(dagilim.keys())
    agirliklar = list(dagilim.values())
    return random.choices(secenekler, weights=agirliklar, k=1)[0]


def veri_seti_olustur(satir_sayisi=5000):
    """Belirtilen sayıda ihbar verisi oluşturur"""
    veriler = []
    
    for _ in range(satir_sayisi):
        # Rastgele olay türü seç
        olay_turu = random.choice(list(OLAY_TURLERI.keys()))
        olay_bilgi = OLAY_TURLERI[olay_turu]
        
        # Konum oluştur
        konum = konum_olustur()
        
        # İhbar metni oluştur
        sablon = random.choice(olay_bilgi["sablonlar"])
        ihbar = sablon.format(konum=konum["konum_str"])
        
        # Öncelik seç
        oncelik = oncelik_sec(olay_bilgi["oncelik_dagilimi"])
        
        # İlgili birim seç
        birim = random.choice(olay_bilgi["birimler"])
        
        veriler.append({
            "ihbar": ihbar,
            "olay_turu": olay_turu,
            "oncelik": oncelik,
            "birim": birim,
            "ilce": konum["ilce"],
            "mahalle": konum["mahalle"],
            "lat": round(konum["lat"], 6),
            "lon": round(konum["lon"], 6)
        })
    
    return pd.DataFrame(veriler)


def main():
    # Veri klasörünü oluştur
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Veri setini oluştur
    print("5000 satırlık veri seti oluşturuluyor...")
    df = veri_seti_olustur(5000)
    
    # CSV olarak kaydet
    csv_path = os.path.join(data_dir, "akom_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    print(f"\nVeri seti başarıyla oluşturuldu: {csv_path}")
    print(f"\nToplam satır sayısı: {len(df)}")
    print(f"\nOlay türü dağılımı:")
    print(df["olay_turu"].value_counts())
    print(f"\nÖncelik dağılımı:")
    print(df["oncelik"].value_counts())
    print(f"\nBirim dağılımı:")
    print(df["birim"].value_counts())
    
    # Örnek veriler göster
    print("\n--- Örnek Veriler ---")
    print(df.head(10).to_string())
    
    return df


if __name__ == "__main__":
    main()
