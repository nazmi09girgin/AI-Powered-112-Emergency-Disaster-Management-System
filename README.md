# AI-Powered-112-Emergency-Disaster-Management-System

## Proje Özeti

Bu proje, **112 Acil Çağrı sistemi** üzerinden gelen **sesli ve yazılı ihbarları** yapay zeka ile analiz ederek,  
afet ve acil durumlarda **doğru müdahale birimlerinin hızlı ve otomatik şekilde yönlendirilmesini** amaçlayan  
**NLP ve LLM tabanlı** bir karar destek sistemidir.

Proje, **İBB Akıllı Şehir Şube Müdürlüğü** ve **Tech Istanbul** tarafından düzenlenen  
**“Yapay Zeka ile Akıllı Şehir Çözümleri Hackathonu”** kapsamında geliştirilip 3.lük elde edilmiştir..

Hackathon; **akıllı ulaşım, akıllı enerji, akıllı çevre, akıllı yaşam ve yeni nesil teknolojilerin kamu hizmetlerine entegrasyonu** temaları etrafında, şehir yaşamını daha **verimli, güvenli ve sürdürülebilir** hâle getirmeyi odaklanan yenilikçi çözümleri teşvik etmeyi amaçlamıştır.

---

## Problem Tanımı

Afet ve kriz anlarında 112’ye gelen ihbarlar:

- Yoğun ve eş zamanlıdır  
- Çoğunlukla **serbest konuşma** ve **doğal dil** içerir  
- Manuel değerlendirme, **zaman kaybına ve hatalı yönlendirmelere** yol açabilir  

Bu durum, özellikle büyük ölçekli afetlerde **müdahale süresini uzatarak** ciddi sonuçlar doğurabilir.

---

## Projenin Amacı

- 112 ihbarlarını **otomatik olarak analiz etmek**
- Olay türünü ve aciliyet seviyesini **yapay zeka teknikleri ile belirlemek**
- İlgili birimlerin (**AKOM, AFAD, İtfaiye, Sağlık, Emniyet**)  
  **doğru ve hızlı şekilde yönlendirilmesini sağlamak**
- Kriz anlarında **zaman ve kaynak kullanımını optimize etmek**

---

## Yapay Zeka Yaklaşımı

### Sesli İhbar İşleme

- **OpenAI Whisper (Large)** modeli kullanılmıştır
- Türkçe konuşma tanıma performansı yüksek olduğu için tercih edilmiştir
- 112’ye gelen sesli çağrılar otomatik olarak **metne dönüştürülür**

---

### Metin Analizi ve Olay Sınıflandırma (NLP)

Elde edilen ihbar metni üzerinde aşağıdaki işlemler gerçekleştirilir:

- **Olay Türü Tespiti**  
  (Deprem, Yangın, Sel, Trafik Kazası vb.)
- **Öncelik Seviyesi Belirleme**  
  (Kritik / Yüksek / Orta)
- **İlçe ve konum bilgisi çıkarımı**
- **Müdahale birimi eşleştirme**

Bu aşamada transformer tabanlı NLP modelleri kullanılmıştır.

#### Kullanılan Model

- **dbmdz/bert-base-turkish-cased**  
  Hugging Face:  https://huggingface.co/dbmdz/bert-base-turkish-cased

Bu model, Türkçe metinlerde anlamsal bağlamı güçlü şekilde yakalayabildiği için  
olay sınıflandırma ve aciliyet tespiti süreçlerinde kullanılmıştır.

---

## Sistem Akışı

1. 112 üzerinden gelen sesli ihbar
2. Whisper ile **Speech-to-Text**
3. BERT tabanlı **NLP analizi**
4. Olay türü ve öncelik tespiti
5. Konum ve birim eşleştirme
6. Müdahale biriminin yönlendirilmesi
7. Kayıt ve analiz çıktılarının oluşturulması

---

## Kullanılan Teknolojiler

- **Python**
- **OpenAI Whisper (Large)**
- **BERT / Transformer tabanlı NLP**
- **dbmdz/bert-base-turkish-cased**
- **Pandas**
- **CSV tabanlı veri kayıt yapıları**
- **Konum & koordinat eşleştirme mekanizmaları (OpenStreetMap)**

---

## Afet ve Akıllı Şehir Katkısı

Bu sistem, afet anlarında:

- İhbarların **manuel değerlendirme yükünü azaltır**
- Müdahale süresini **kısaltır**
- Karar alma süreçlerini **standartlaştırır**
- Yapay zekayı, **akıllı şehir ve kamu hizmetleri** bağlamında  
  gerçek hayatta uygulanabilir bir çözüme dönüştürür

---

## Kullanım Senaryoları

- Deprem sonrası yoğun 112 çağrıları
- Sel ve su baskını ihbarları
- Büyük çaplı yangınlar
- Trafik kazaları ve çoklu acil durumlar

Sistem, kriz anlarında **ölçeklenebilir ve otomatik** bir yapı sunar.

---

## 🏁 Sonuç

Bu proje, **112 Acil Çağrı sistemi** ile entegre çalışan,  
**yapay zeka destekli** bir afet yönetimi çözümü sunar.

Hedef; teknolojiyi, **doğru karar, hızlı müdahale, minimum kayıp**  
ilkeleri doğrultusunda kullanmaktır.
