import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import AKOMClassifier, OLAY_TURLERI, ONCELIKLER, BIRIMLER
from src.utils import get_ilce_koordinat, load_dataset, get_priority_color, get_event_icon, save_analysis
import requests
import time

try:
    from audio_recorder_streamlit import audio_recorder
    from src.speech import transcribe_audio, is_whisper_available
    VOICE_INPUT_AVAILABLE = True
except ImportError:
    VOICE_INPUT_AVAILABLE = False

def geocode_address(mahalle=None, cadde=None, ilce=None, city="İstanbul"):
    """OpenStreetMap Nominatim API ile adres koordinatları alındı"""
    try:
        address_variants = []
        
        if cadde and mahalle and ilce:
            address_variants.append(f"{cadde} Caddesi, {mahalle} Mahallesi, {ilce}, {city}")
        if cadde and ilce:
            address_variants.append(f"{cadde} Caddesi, {ilce}, {city}")
        if mahalle and ilce:
            address_variants.append(f"{mahalle} Mahallesi, {ilce}, {city}")
        if mahalle:
            address_variants.append(f"{mahalle}, {ilce or city}")
        
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "AKOM-Decision-Support/1.0"}
        
        for address in address_variants:
            params = {
                "q": f"{address}, Türkiye",
                "format": "json",
                "limit": 1,
                "countrycodes": "tr"
            }
            response = requests.get(url, params=params, headers=headers, timeout=5)
            
            if response.status_code == 200 and response.json():
                data = response.json()[0]
                return {
                    "lat": float(data["lat"]),
                    "lon": float(data["lon"]),
                    "display_name": data.get("display_name", address)
                }
            time.sleep(0.5)  # Rate limiting
            
    except Exception as e:
        print(f"Geocoding hatası: {e}")
    
    return None


def find_nearest_emergency_center(lat, lon, center_type="fire_station"):
    """
    OpenStreetMap Overpass API ile en yakın acil müdahale merkezini bul
    center_type: fire_station, hospital, police
    """
    import math
    
    # Overpass API query
    type_mapping = {
        "fire_station": "amenity=fire_station",
        "hospital": "amenity=hospital",
        "police": "amenity=police"
    }
    
    type_names = {
        "fire_station": "İtfaiye",
        "hospital": "Hastane",
        "police": "Polis Merkezi"
    }
    
    amenity = type_mapping.get(center_type, "amenity=fire_station")
    
    # 10km yarıçapında ara
    query = f"""
    [out:json][timeout:10];
    (
      node[{amenity}](around:10000,{lat},{lon});
      way[{amenity}](around:10000,{lat},{lon});
    );
    out center;
    """
    
    try:
        url = "https://overpass-api.de/api/interpreter"
        response = requests.post(url, data={"data": query}, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            elements = data.get("elements", [])
            
            if not elements:
                return None
            
            # En yakın olanı bul
            nearest = None
            min_distance = float('inf')
            
            for elem in elements:
                elem_lat = elem.get("lat") or elem.get("center", {}).get("lat")
                elem_lon = elem.get("lon") or elem.get("center", {}).get("lon")
                
                if elem_lat and elem_lon:
                    # Haversine mesafe hesaplama
                    R = 6371  # km
                    dlat = math.radians(elem_lat - lat)
                    dlon = math.radians(elem_lon - lon)
                    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat)) * math.cos(math.radians(elem_lat)) * math.sin(dlon/2)**2
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    distance = R * c
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest = {
                            "lat": elem_lat,
                            "lon": elem_lon,
                            "name": elem.get("tags", {}).get("name", type_names[center_type]),
                            "distance_km": round(distance, 2),
                            "type": type_names[center_type]
                        }
            
            return nearest
            
    except Exception as e:
        print(f"Overpass API hatası: {e}")
    
    return None

# Sayfa ayarları
st.set_page_config(
    page_title="Afet Koordinasyon Sistemi",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Ana sayfa arka planı */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
    }
    
    /* Modern Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
        box-shadow: 4px 0 20px rgba(0,0,0,0.3);
    }
    
    [data-testid="stSidebar"]::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at top left, rgba(99,179,237,0.15) 0%, transparent 50%);
        pointer-events: none;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: white !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    /* Menü başlığı */
    section[data-testid="stSidebar"] h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: white !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1.5rem;
    }
    
    /* Radio butonları - Modern stil */
    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
        color: white !important;
        font-weight: 500;
    }
    
    [data-testid="stSidebar"] .stRadio label span,
    [data-testid="stSidebar"] .stRadio label p {
        color: white !important;
        font-size: 1rem;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(5px);
    }
    
    [data-testid="stSidebar"] .stRadio div[data-checked="true"] label {
        background: rgba(255,255,255,0.2);
        border-left: 3px solid #4fd1c5;
    }
    
    /* Ana içerik metinleri */
    .stApp, .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #1a365d !important;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(90deg, #E8F4FD 0%, #B8D4E8 100%);
        border-radius: 10px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .kritik-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2E5A8F 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .yuksek-card {
        background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .orta-card {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .result-box {
        background: #FFFFFF;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E3A5F;
        margin: 1rem 0;
        color: #1E3A5F;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .result-box h4 {
        color: #1E3A5F;
        margin-bottom: 1rem;
    }
    
    .result-box p {
        color: #1E3A5F;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1E3A5F 0%, #2E5A8F 100%);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2E5A8F 0%, #1E3A5F 100%);
        color: white !important;
    }
    
    .stButton > button p, .stButton > button span {
        color: white !important;
    }
    
    /* Metric kartları */
    [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #2E5A8F !important;
    }
    
    /* TextArea ve SelectBox */
    .stTextArea textarea, .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        border: 2px solid #B8D4E8 !important;
        color: #1E3A5F !important;
    }
    
    /* Multiselect yazıları */
    .stMultiSelect > div > div {
        background-color: #FFFFFF !important;
        color: #1E3A5F !important;
    }
    
    .stMultiSelect span {
        color: #1E3A5F !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #1E3A5F !important;
        color: white !important;
    }
    
    .stMultiSelect [data-baseweb="tag"] span {
        color: white !important;
    }
    
    /* Slider */
    .stSlider label, .stSlider p {
        color: #1E3A5F !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        color: #1E3A5F !important;
        background-color: #E8F4FD !important;
    }
    
    /* Info box */
    .stAlert {
        background-color: #E8F4FD !important;
        color: #1E3A5F !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classifier():
    return AKOMClassifier(use_bert=False)  # Hızlı başlangıç için kural tabanlı


@st.cache_data
def load_data():
    return load_dataset()


def create_map(lat, lon, popup_text, priority="Orta", nearest_center=None):
    m = folium.Map(
        location=[lat, lon],
        zoom_start=14,
        tiles="OpenStreetMap"
    )
    
    color_map = {
        "Kritik": "red",
        "Yüksek": "orange",
        "Orta": "blue"
    }
    
    # Olay noktası
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_text, max_width=300),
        icon=folium.Icon(color=color_map.get(priority, "blue"), icon="exclamation-sign", prefix="glyphicon")
    ).add_to(m)
    
    folium.Circle(
        [lat, lon],
        radius=500,
        color=color_map.get(priority, "blue"),
        fill=True,
        fill_opacity=0.2
    ).add_to(m)
    
    # En yakın acil merkez
    if nearest_center:
        center_popup = f"""
        <b>{nearest_center['type']}</b><br>
        {nearest_center['name']}<br>
        <b>Uzaklık:</b> {nearest_center['distance_km']} km
        """
        folium.Marker(
            [nearest_center['lat'], nearest_center['lon']],
            popup=folium.Popup(center_popup, max_width=250),
            icon=folium.Icon(color="green", icon="home", prefix="glyphicon")
        ).add_to(m)
        
        # Olay ile merkez arası çizgi
        folium.PolyLine(
            [[lat, lon], [nearest_center['lat'], nearest_center['lon']]],
            color="green",
            weight=2,
            opacity=0.8,
            dash_array="5, 10"
        ).add_to(m)
    
    return m


def create_overview_map(df_sample):
    m = folium.Map(
        location=[41.0082, 28.9784],  # İstanbul merkez
        zoom_start=10,
        tiles="OpenStreetMap"
    )
    
    color_map = {
        "Kritik": "red",
        "Yüksek": "orange",
        "Orta": "blue"
    }
    
    icon_map = {
        "Kritik": "exclamation-sign",
        "Yüksek": "warning-sign", 
        "Orta": "info-sign"
    }
    
    for _, row in df_sample.iterrows():
        popup_html = f"""
        <div style='width:200px'>
            <b>Olay:</b> {row['olay_turu']}<br>
            <b>Öncelik:</b> {row['oncelik']}<br>
            <b>İlçe:</b> {row['ilce']}<br>
            <b>Birim:</b> {row.get('birim', 'N/A')}
        </div>
        """
        
        folium.Marker(
            [row['lat'], row['lon']],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['olay_turu']} - {row['oncelik']}",
            icon=folium.Icon(
                color=color_map.get(row['oncelik'], "blue"),
                icon=icon_map.get(row['oncelik'], "info-sign"),
                prefix="glyphicon"
            )
        ).add_to(m)
    
    return m


def main():
    st.markdown('<div class="main-header">Afet Koordinasyon Sistemi</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>AI Destekli İhbar Analiz Platformu</p>", unsafe_allow_html=True)
    
    classifier = load_classifier()
    
    df = load_data()
    
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []
    
    with st.sidebar:
        st.markdown("<h1 style='color: white; text-align: center;'>AKS</h1>", unsafe_allow_html=True)
        
        menu = st.radio(
            "Menü",
            ["İhbar Analizi", "Geçmiş", "İstatistikler", "Olay Haritası", "Veri Seti"],
            label_visibility="collapsed"
        )
    
    if menu == "İhbar Analizi":
        st.header("Yeni İhbar Analizi")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if 'selected_ihbar' not in st.session_state:
                st.session_state['selected_ihbar'] = ""
            
            if VOICE_INPUT_AVAILABLE:
                st.markdown("### Sesli İhbar Girişi")
                st.caption("Mikrofona tıklayıp ihbarınızı söyleyin, bırakınca metin olarak görünecek.")
                
                if 'last_audio_hash' not in st.session_state:
                    st.session_state['last_audio_hash'] = None
                
                audio_bytes = audio_recorder(
                    text="",
                    recording_color="#e74c3c",
                    neutral_color="#1E3A5F",
                    icon_name="microphone",
                    icon_size="2x",
                    pause_threshold=2.0,
                    sample_rate=16000
                )
                
                if audio_bytes:
                    import hashlib
                    audio_hash = hashlib.md5(audio_bytes).hexdigest()
                    
                    if audio_hash != st.session_state.get('last_audio_hash'):
                        with st.spinner("Ses metne çevriliyor..."):
                            transcribed_text = transcribe_audio(audio_bytes)
                            if transcribed_text:
                                st.session_state['selected_ihbar'] = transcribed_text
                                st.session_state['last_audio_hash'] = audio_hash
                                st.success("Ses başarıyla metne çevrildi!")
                            else:
                                st.error("Ses metne çevrilemedi. Lütfen tekrar deneyin.")
                    else:
                        if st.session_state.get('selected_ihbar'):
                            st.success("Ses zaten metne çevrildi, aşağıda görebilirsiniz.")
                
                st.markdown("---")
            
            with st.expander("Örnek İhbarlar", expanded=False):
                example_reports = [
                    "Kadıköy Fenerbahçe Mahallesi'nde yoğun yağış nedeniyle sel baskını yaşanıyor. Alt geçitler su altında kaldı.",
                    "Avcılar Cihangir Mahallesi'nde şiddetli deprem hissedildi. Binalarda hasar var, ekiplere hemen yönlendirilsin.",
                    "Bakırköy sahil yolunda deprem sonrası çatlaklar oluştu. Bina boşaltılması gerekiyor.",
                    "Esenyurt'ta aşırı yağış nedeniyle dere taştı. Evlere su girdi, mahsur kalan vatandaşlar var.",
                    "Sarıyer ormanlık alanda büyük yangın çıktı. Alevler rüzgarla hızla yayılıyor.",
                    "Beşiktaş Levent'te yoğun gaz kokusu alınıyor. Patlama riski var, acil müdahale gerekli.",
                    "Pendik'te deprem sonrası bina çökmesi. Enkazda mahsur kalanlar olabilir.",
                    "Beylikdüzü'nde sel suları araçları sürükledi. E-5 karayolu trafiğe kapandı."
                ]
                
                for i, example in enumerate(example_reports):
                    if st.button(example[:50] + "...", key=f"example_{i}", use_container_width=True):
                        st.session_state['selected_ihbar'] = example
                        st.rerun()
            
            ihbar_text = st.text_area(
                "İhbar Metnini Girin (veya yukarıdan sesle girin)",
                value=st.session_state.get('selected_ihbar', ''),
                height=150,
                placeholder="Örnek: Avcılar Cihangir Mahallesi Cumhuriyet Caddesi yakınlarında şiddetli deprem hissedildi, binalarda hasar var."
            )
            
            analyze_btn = st.button("Analiz Et", use_container_width=True)
        
        with col2:
            st.markdown("### Bilgi")
            st.info("""
            **Sistem Özellikleri:**
            - Sesli ihbar girişi (Whisper AI)
            - AI tabanlı sınıflandırma
            - Otomatik konum tespiti
            - Öncelik belirleme
            - Birim yönlendirme
            """)
        
        if analyze_btn and ihbar_text:
            with st.spinner("İhbar analiz ediliyor..."):
                result = classifier.analyze(ihbar_text)
            
            # Yinelenen ihbar kontrolü
            is_duplicate = False
            duplicate_entry = None
            for entry in st.session_state.get('analysis_history', []):
                # Aynı ilçe ve olay türü kontrolü
                if (entry.get('ilce') == result.get('ilce') and 
                    entry.get('olay_turu') == result.get('olay_turu')):
                    is_duplicate = True
                    duplicate_entry = entry
                    break
            
            if is_duplicate and duplicate_entry:
                st.warning(f"""
                **Bu bölgede aynı türde bir olayla zaten ilgileniliyor!**
                
                - **Önceki kayıt:** {duplicate_entry['tarih']}
                - **Olay türü:** {duplicate_entry['olay_turu']}
                - **Konum:** {duplicate_entry.get('ilce', 'İstanbul')}
                """)
            
            st.session_state['analysis_result'] = result
            st.session_state['analyzed_text'] = ihbar_text
            
            from datetime import datetime
            history_entry = {
                'tarih': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'ihbar': ihbar_text[:100] + "..." if len(ihbar_text) > 100 else ihbar_text,
                'ihbar_tam': ihbar_text,
                'olay_turu': result['olay_turu'],
                'oncelik': result['oncelik'],
                'birimler': result.get('birimler', [result['birim']]),
                'ilce': result.get('ilce', 'İstanbul'),
                'result': result
            }
            st.session_state['analysis_history'].insert(0, history_entry)  # En yenisi başa
            
            save_analysis(result, ihbar_text)
            load_data.clear()
        
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            ihbar_text = st.session_state['analyzed_text']
            
            adres_parts = []
            if result.get('bolge'):
                adres_parts.append(result['bolge'])
            if result.get('mahalle'):
                adres_parts.append(f"{result['mahalle']} Mah.")
            if result.get('cadde'):
                adres_parts.append(f"{result['cadde']} Cad.")
            if result.get('sokak'):
                adres_parts.append(f"{result['sokak']} Sok.")
            if result.get('ilce'):
                adres_parts.append(result['ilce'])
            
            tam_adres = ", ".join(adres_parts) if adres_parts else "İstanbul"
            
            col1, col2, col3, col4 = st.columns(4)
            
            birimler_list = result.get('birimler', [result['birim']])
            birimler_text = ", ".join(birimler_list)
            
            with col1:
                st.metric(
                    label="Olay Türü",
                    value=result['olay_turu'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    label="Öncelik",
                    value=result['oncelik'],
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="Ana Birim",
                    value=result['birim'],
                    delta=None
                )
            
            with col4:
                st.metric(
                    label="Konum",
                    value=result['ilce'] or "İstanbul",
                    delta=None
                )
            
            st.markdown("### Olay Konumu")
            st.markdown(f"**Adres:** {tam_adres}")
            
            geocoded = None
            mahalle = result.get('mahalle')
            cadde = result.get('cadde')
            ilce = result.get('ilce')
            
            if mahalle or cadde:
                with st.spinner("Konum belirleniyor..."):
                    geocoded = geocode_address(mahalle=mahalle, cadde=cadde, ilce=ilce)
            
            if geocoded:
                lat = geocoded['lat']
                lon = geocoded['lon']
            elif result.get('ilce'):
                coords = get_ilce_koordinat(result['ilce'])
                lat = coords['lat']
                lon = coords['lon']
            else:
                lat = 41.0082
                lon = 28.9784
            
            popup_text = f"""
            <b>Olay:</b> {result['olay_turu']}<br>
            <b>Öncelik:</b> {result['oncelik']}<br>
            <b>Birim:</b> {result['birim']}<br>
            <b>Adres:</b> {tam_adres}
            """
            
            # En yakın acil müdahale merkezini bul (session_state ile cache)
            cache_key = f"nearest_center_{lat}_{lon}_{result['birim']}"
            
            if cache_key not in st.session_state:
                birim = result['birim']
                olay = result['olay_turu']
                
                # Birime ve olay türüne göre merkez türü belirle
                if "İtfaiye" in birim or "Yangın" in olay:
                    center_type = "fire_station"
                elif "Sağlık" in birim or "Ambulans" in birim or "Deprem" in olay:
                    center_type = "hospital"  # Deprem için hastane
                else:
                    center_type = "fire_station"  # Varsayılan
                
                with st.spinner("En yakın müdahale merkezi aranıyor..."):
                    st.session_state[cache_key] = find_nearest_emergency_center(lat, lon, center_type)
            
            nearest_center = st.session_state.get(cache_key)
            
            # En yakın merkez bilgisi
            if nearest_center:
                st.success(f"En yakın {nearest_center['type']}: **{nearest_center['name']}** ({nearest_center['distance_km']} km)")
            
            m = create_map(lat, lon, popup_text, result['oncelik'], nearest_center)
            st_folium(m, use_container_width=True, height=400, key="analiz_haritasi", returned_objects=[])
            
            st.markdown("### Detaylı Analiz Raporu")
            st.markdown(f"""
            <div class="result-box">
                <h4>Olay Detayları</h4>
                <p><b>İhbar:</b> {ihbar_text}</p>
                <p><b>Tespit Edilen Olay:</b> {result['olay_turu']}</p>
                <p><b>Öncelik Seviyesi:</b> {result['oncelik']}</p>
                <p><b>Yönlendirilecek Birimler:</b> {birimler_text}</p>
                <p><b>Konum:</b> {tam_adres}</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("Aktif Öğrenme - Sonucu Düzelt", expanded=False):
                st.markdown("**Sonuç yanlış mı? Doğru değerleri seçerek modeli eğitin:**")
                
                col_fb1, col_fb2 = st.columns(2)
                
                with col_fb1:
                    correct_olay = st.selectbox(
                        "Doğru Olay Türü",
                        OLAY_TURLERI + ["Diğer"],
                        index=OLAY_TURLERI.index(result['olay_turu']) if result['olay_turu'] in OLAY_TURLERI else len(OLAY_TURLERI)
                    )
                    
                with col_fb2:
                    correct_oncelik = st.selectbox(
                        "Doğru Öncelik",
                        ONCELIKLER,
                        index=ONCELIKLER.index(result['oncelik']) if result['oncelik'] in ONCELIKLER else 0
                    )
                
                if st.button("Geri Bildirimi Kaydet", key="feedback_btn"):
                    import os
                    from datetime import datetime
                    
                    feedback_path = os.path.join(os.path.dirname(__file__), "data", "feedback.csv")
                    feedback_data = {
                        'tarih': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'ihbar': ihbar_text,
                        'tahmin_olay': result['olay_turu'],
                        'dogru_olay': correct_olay,
                        'tahmin_oncelik': result['oncelik'],
                        'dogru_oncelik': correct_oncelik
                    }
                    
                    import pandas as pd
                    if os.path.exists(feedback_path):
                        fb_df = pd.read_csv(feedback_path)
                        fb_df = pd.concat([fb_df, pd.DataFrame([feedback_data])], ignore_index=True)
                    else:
                        fb_df = pd.DataFrame([feedback_data])
                    
                    fb_df.to_csv(feedback_path, index=False)
                    st.success("Geri bildirim kaydedildi! Teşekkürler, bu veriler modelin geliştirilmesinde kullanılacak.")
    
    elif menu == "Geçmiş":
        st.header("Analiz Geçmişi")
        
        history = st.session_state.get('analysis_history', [])
        
        if len(history) == 0:
            st.info("Henüz analiz yapılmamış. İhbar Analizi sayfasından yeni bir analiz yapın.")
        else:
            st.write(f"**Toplam {len(history)} analiz kaydı**")
            
            # Geçmiş tablosu
            for i, entry in enumerate(history):
                with st.expander(f"{entry['tarih']} - {entry['olay_turu']} ({entry['oncelik']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**İhbar:** {entry['ihbar_tam']}")
                        st.markdown(f"**Olay Türü:** {entry['olay_turu']}")
                        st.markdown(f"**Öncelik:** {entry['oncelik']}")
                    with col2:
                        st.markdown(f"**Birimler:** {', '.join(entry['birimler'])}")
                        st.markdown(f"**Konum:** {entry.get('ilce', 'İstanbul')}")
            
            # Geçmişi temizle butonu
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("Geçmişi Temizle"):
                    st.session_state['analysis_history'] = []
                    st.rerun()
            
            with col_btn2:
                import pandas as pd
                export_data = []
                for entry in history:
                    export_data.append({
                        'Tarih': entry['tarih'],
                        'İhbar': entry['ihbar_tam'],
                        'Olay Türü': entry['olay_turu'],
                        'Öncelik': entry['oncelik'],
                        'Birimler': ', '.join(entry['birimler']),
                        'Konum': entry.get('ilce', 'İstanbul')
                    })
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="CSV Olarak İndir",
                    data=csv,
                    file_name="analiz_gecmisi.csv",
                    mime="text/csv"
                )
    
    elif menu == "İstatistikler":
        st.header("Veri Seti İstatistikleri")
        
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam İhbar", f"{len(df):,}")
            
            with col2:
                kritik = len(df[df['oncelik'] == 'Kritik'])
                st.metric("Kritik", f"{kritik:,}")
            
            with col3:
                yuksek = len(df[df['oncelik'] == 'Yüksek'])
                st.metric("Yüksek", f"{yuksek:,}")
            
            with col4:
                orta = len(df[df['oncelik'] == 'Orta'])
                st.metric("Orta", f"{orta:,}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Olay Türü Dağılımı")
                olay_counts = df['olay_turu'].value_counts()
                st.bar_chart(olay_counts)
            
            with col2:
                st.subheader("Birim Dağılımı")
                birim_counts = df['birim'].value_counts()
                st.bar_chart(birim_counts)
            
            st.subheader("İlçe Bazlı Dağılım")
            ilce_counts = df['ilce'].value_counts().head(15)
            st.bar_chart(ilce_counts)
        else:
            st.warning("Veri seti bulunamadı. Lütfen önce veri setini oluşturun.")
    
    elif menu == "Olay Haritası":
        st.header("İstanbul Olay Haritası")
        
        if df is not None:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.subheader("Filtreler")
                
                selected_priority = st.multiselect(
                    "Öncelik",
                    ["Kritik", "Yüksek", "Orta"],
                    default=["Kritik", "Yüksek"]
                )
                
                selected_types = st.multiselect(
                    "Olay Türü",
                    df['olay_turu'].unique().tolist(),
                    default=df['olay_turu'].unique().tolist()[:3]
                )
                
                sample_size = st.slider("Gösterilecek Kayıt", 10, 200, 50)
                
                show_heatmap = st.checkbox("Isı Haritası Göster", value=False)
            
            with col1:
                filtered_df = df[
                    (df['oncelik'].isin(selected_priority)) &
                    (df['olay_turu'].isin(selected_types))
                ].head(sample_size)
                
                if show_heatmap:
                    from folium.plugins import HeatMap
                    m = folium.Map(
                        location=[41.0082, 28.9784],
                        zoom_start=10,
                        tiles="OpenStreetMap"
                    )
                    heat_data = [[row['lat'], row['lon']] for _, row in filtered_df.iterrows()]
                    HeatMap(heat_data, radius=15, blur=10).add_to(m)
                    st_folium(m, width=800, height=500, key="olay_heatmap")
                    st.markdown("**Yoğunluk haritası - Kırmızı alanlar daha fazla olay**")
                else:
                    m = create_overview_map(filtered_df)
                    st_folium(m, width=800, height=500, key="olay_haritasi")
                    st.markdown("**Kritik (Kırmızı) | Yüksek (Turuncu) | Orta (Mavi)**")
        else:
            st.warning("Veri seti bulunamadı.")
    
    elif menu == "Veri Seti":
        st.header("Veri Seti Görüntüleyici")
        
        if df is not None:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_olay = st.selectbox("Olay Türü", ["Tümü"] + df['olay_turu'].unique().tolist())
            
            with col2:
                filter_oncelik = st.selectbox("Öncelik", ["Tümü"] + df['oncelik'].unique().tolist())
            
            with col3:
                filter_ilce = st.selectbox("İlçe", ["Tümü"] + sorted(df['ilce'].unique().tolist()))
            
            # Filtreleme
            filtered_df = df.copy()
            if filter_olay != "Tümü":
                filtered_df = filtered_df[filtered_df['olay_turu'] == filter_olay]
            if filter_oncelik != "Tümü":
                filtered_df = filtered_df[filtered_df['oncelik'] == filter_oncelik]
            if filter_ilce != "Tümü":
                filtered_df = filtered_df[filtered_df['ilce'] == filter_ilce]
            
            st.write(f"**Toplam: {len(filtered_df)} kayıt**")
            st.dataframe(filtered_df, use_container_width=True, height=500)
            
            # CSV indirme
            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "CSV İndir",
                csv,
                "akom_filtered_data.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("Veri seti bulunamadı.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #888;'>© 2024 Afet Koordinasyon Sistemi | AI-Powered Emergency Response Platform</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
