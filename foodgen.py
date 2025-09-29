import streamlit as st
import random
import requests
import json
from io import BytesIO
from transformers import pipeline
from PIL import Image
import io

# BLIP

# Inisialisasi Pipeline untuk BLIP (Model Captioning)
@st.cache_resource
def load_blip_pipeline():
    """Memuat model BLIP (Image Captioning) dari Hugging Face secara lokal."""
    st.info("⏳ Tunggu sebentar sedang dijalankan")
    try:
        # Gunakan 'image-to-text' task dan model BLIP
        # device=-1 berarti menggunakan CPU.
        pipe = pipeline(
            "image-to-text", 
            model="Salesforce/blip-image-captioning-base",
            device=-1 
        )
        st.success("✅ Model BLIP berhasil dimuat.")
        return pipe
    except Exception as e:
        # Jika gagal (biasanya karena ukuran model), pakai dummy
        st.error(f"Gagal memuat pipeline BLIP (kemungkinan masalah memori/dependency): {e}")
        st.info("Aplikasi akan menggunakan deskripsi DUMMY sebagai gantinya.")
        return None

# Muat pipeline saat aplikasi berjalan
blip_pipeline = load_blip_pipeline()

# Class untuk Food Generator
class FoodGenerator:
    """Class untuk mengelola logika rekomendasi makanan."""
    
    def __init__(self, food_data):
        self.food_options = food_data

    def generate_menu_from_ingredients(self, ingredients_list, budget):
        """Logika untuk mendapatkan saran resep/makanan dari daftar bahan."""
        if not ingredients_list:
            return "Silakan unggah foto untuk mendapatkan referensi masakan!"

        bahan_str = ", ".join(ingredients_list)
        
        # untuk budget
        prompt = (
            f"Berdasarkan bahan-bahan berikut: {bahan_str}. "
            f"Budget maksimal untuk beli bahan pelengkap adalah Rp {budget:,}. "
            f"Sebagai anak kos, berikan 3 ide makanan yang simpel, mudah dibuat, dan MURAH. "
            f"Berikan nama makanan dan perkiraan langkah singkat. "
            f"Balas dalam format list bernomor."
        )
        
        system_message = (
            "Kamu adalah Asisten Resep Anak Kos. Tugasmu adalah memberikan ide "
            "masakan yang praktis dari bahan-bahan yang disebutkan."
        )
        
        messages_payload = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
        
        # Menggunakan OpenRouter API->pakai grok-4-fast
        return get_ai_response(messages_payload, "x-ai/grok-4-fast:free")

# respons dari OpenRouter
def get_ai_response(messages_payload, model):
    """Fungsi untuk memanggil OpenRouter API (digunakan untuk ide resep)."""
    # Ganti dengan API Key milikmu yang valid
    api_key = "OPENROUTER_API_KEY"

    try:
        # Panggil API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            data=json.dumps({"model": model, "messages": messages_payload, "max_tokens": 1000, "temperature": 0.7})
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        # Jika API Key OpenRouter bermasalah, berikan respons dummy
        st.error(f"Error memanggil OpenRouter API: {e}")
        st.info("Pastikan API Key OpenRouter Anda valid.")
        
        # Respon dummy untuk generator resep jika API eksternal gagal
        if messages_payload[-1]["role"] == "user" and "Berdasarkan bahan-bahan berikut" in messages_payload[-1]["content"]:
            return "1. **Nasi dan Telur Ceplok:** Bahan utama sudah lengkap. Cukup goreng telur, beri garam, dan sajikan dengan nasi panas. \n2. **Mie Instan Simpel:** Rebus mie instan dengan bumbu, sangat cepat. \n3. **Tumis Sayur Cepat:** Jika ada sawi/bayam, tumis dengan sedikit bawang dan garam."
        
        return "Error saat mendapatkan respons AI."
    except json.JSONDecodeError:
        st.error("Error: Respons API tidak valid.")
        return "Error: Respons API tidak valid."


# image to text dengan BLIP

def get_image_caption_hf(image_bytes):
    """
    Fungsi utama Image-to-Text. Prioritas: 
    1. BLIP Lokal
    2. Fallback Dummy
    """
    
    # dummy fallback jika BLIP gagal
    def fallback_dummy():
        st.warning("⚠️ Menggunakan deskripsi DUMMY karena model BLIP tidak dimuat. Akurasi terbatas.")
        dummy_captions = {
            "mie_instan": ("mie instan, telur, sawi, bumbu dapur", ["mie instan", "telur", "sawi", "bumbu"]),
            "nasi_ayam": ("nasi, ayam mentah, cabai, bawang", ["nasi", "ayam", "cabai", "bawang"]),
            "sayuran": ("sayuran hijau, tomat, wortel", ["sayuran hijau", "tomat", "wortel"]),
            "default": ("bahan makanan acak, seperti sayuran dan telur", ["sayuran", "telur"])
        }
        caption_key = random.choice(list(dummy_captions.keys()))
        caption, bahan_list = dummy_captions[caption_key]
        return caption, bahan_list

    #1. --- Pilihan 1: BLIP Lokal ---
    if blip_pipeline is not None:
        st.info("⏳ Memproses gambar dengan BLIP (Lokal)...")
        try:
            # Konversi bytes gambar dari Streamlit ke objek Image (PIL)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Panggil pipeline untuk melakukan inferensi
            result = blip_pipeline(image)
            
            if result and isinstance(result, list) and 'generated_text' in result[0]:
                caption = result[0]['generated_text']
                
                # Memproses hasil caption (misalnya: "a plate of eggs and spinach")
                # Kita lakukan pemrosesan sederhana menjadi list bahan
                # Hanya mengambil kata-kata yang deskriptif (bukan kata sandang atau kata kerja)
                bahan_list = [
                    item.strip() 
                    for item in caption.split(' ') 
                    if item.strip() and len(item.strip()) > 2 and item.lower() not in ["a", "the", "of", "in", "and", "with", "is", "some"]
                ]

                st.success("✅ Deskripsi gambar berhasil didapatkan dari BLIP.")
                return caption, bahan_list
            
            # Jika BLIP gagal memberikan hasil teks
            st.warning("Model BLIP gagal mendeskripsikan gambar, menggunakan dummy.")
            return fallback_dummy()

        except Exception as e:
            st.error(f"Error saat inferensi BLIP lokal: {e}")
            return fallback_dummy()
            
    # --- Pilihan 2: Fallback dummy---
    else:
        return fallback_dummy()


# --- 5. Konfigurasi Streamlit & Initial State ---
st.set_page_config(page_title="Anak Kos Food Generator", page_icon="🍴", layout="centered")

# background color & chat message style
st.markdown(
    """
    <style>
    /* 1. WARNA BACKGROUND UTAMA & SIDEBAR (Pink Muda) */
    .stApp {
        background-color: #F8BBD0; 
    }
    [data-testid="stSidebar"] {
        background-color: #F8BBD0 !important; 
    }
    .main .block-container {
        background-color: transparent; 
    }
    
    /* 2. PESAN PENGGUNA DAN BOT (Putih) */
    .stChatMessage [data-testid="stChatMessageContent"] {
        background-color: white !important; 
        color: #333333 !important; 
        border-radius: 0.5rem;
        padding: 10px 15px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1); 
    }
    /* Pesan bot putih */
    .stChatMessage:nth-child(even) [data-testid="stChatMessageContent"] {
        background-color: white !important;
        color: #333333 !important;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    /* Agar teks di sidebar lebih gelap/jelas (Opsional) */
    .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #333333; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍴 Food Generator Anak Kos")
st.write("Kamu bingung mau makan apa hari ini? Pilih opsi di side bar mau masak atau beli ya! ")

# Database makanan dan Inisialisasi Generator
food_data = {"Asin/Pedas": ["Bebek goreng", "Ayam Geprek"], "Manis": ["Martabak Manis", "Donat"], "Ringan/Sehat": ["Salad Buah", "Salad Sayur"], "Cepat/Simple": ["Mie Instan", "Roti Bakar"]}
generator = FoodGenerator(food_data)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Halo anak kos! Aku siap bantu. Pilih mode **Masak** atau **Beli** di sidebar ya!"})
if "mode" not in st.session_state:
    st.session_state.mode = 'Beli' # Default mode ke Beli
if "user_location" not in st.session_state:
    st.session_state.user_location = "" # Dihapus dari sidebar, tapi disimpan untuk fitur lain
if "budget_choice" not in st.session_state:
    st.session_state.budget_choice = 30000
    
# Sidebar pilih mode beli atau masak
with st.sidebar:
    st.title("⚙️ Pengaturan & Mode")
    
    # Selectbox untuk menentukan mode utama
    selected_mode = st.selectbox(
        "Pilih Mode Utama:", 
        options=['Masak (Upload Bahan)', 'Beli (Chatbot Rekomendasi)'], 
        index=0 if st.session_state.mode == 'Masak' else 1
    )
    # Update session state mode berdasarkan selectbox
    st.session_state.mode = 'Masak' if selected_mode == 'Masak (Upload Bahan)' else 'Beli'
    
    st.markdown("---")
    st.subheader("Pengaturan Mode Beli 🛍️")
    
    # Slider Budget
    max_budget = 100000
    st.session_state.budget_choice = st.slider(
        "💰 Budget Makan Hari Ini (Rupiah)",
        min_value=0, max_value=max_budget, value=st.session_state.budget_choice, step=5000, format='Rp %d'
    )
    
    # Opsi Cepat
    st.markdown("---")
    st.subheader("🍽️ Contoh Opsi")
    for category, items in generator.food_options.items():
        st.markdown(f"**{category}**")
        st.text(f"  - {', '.join(items[:2])}...")

# Tampilan pilihan

if st.session_state.mode == 'Masak':
    st.subheader("🍳 Mode: Masak (Ide Resep dari Bahan)")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Foto Bahan Makanan (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Foto Bahan Makanan Anda", use_column_width=True)
        
        if st.button("🔍 Cari Ide Masakan", key="btn_masak"):
            image_bytes = uploaded_file.getvalue() 
            
            # Panggil fungsi BLIP/Dummy
            caption, ingredients_list = get_image_caption_hf(image_bytes)
            
            st.markdown(f"**Deskripsi Gambar (Perkiraan Bahan):** `{caption}`")
            
            # Jika deskripsi gagal, jangan lanjutkan ke generator
            if "Error" in caption or "Gagal" in caption:
                st.error("Proses pembuatan ide resep dibatalkan karena deskripsi gambar gagal.")
            else:
                with st.spinner("Mencari resep terbaik buat anda..."):
                    resep_text = generator.generate_menu_from_ingredients(ingredients_list, st.session_state.budget_choice)
                
                st.success("✅ Ide Masakan Berhasil Dibuat!")
                st.markdown("---")
                st.subheader("📝 Ide Resep untuk Anda:")
                st.markdown(resep_text)
                st.balloons()
        
                
elif st.session_state.mode == 'Beli':
    st.subheader("🛍️ Mode: Beli (Chatbot Rekomendasi)")
    st.info(f"Rekomendasi disesuaikan dengan **Budget Rp {st.session_state.budget_choice:,}**")
    
    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input (Chat Input)
    if prompt := st.chat_input("Tanyakan rekomendasi, mood, atau preferensi rasa..."):
        
        # 1. Prompt dari user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Respons dari AI
        with st.chat_message("assistant"):
            with st.spinner("Sedang memikirkan rekomendasi terbaik..."):
                
                messages_for_api = st.session_state.messages.copy()
                
                # untuk budget 
                system_prompt_with_context = (
                    "Kamu adalah Asisten Rekomendasi Makanan Beli untuk anak kos. "
                    "Jawab dengan ramah, dan fokus pada memberikan saran makanan yang mudah dibeli (Gojek/GrabFood/Warung). "
                    f"Harga makanan maksimal: **Rp {st.session_state.budget_choice:,}**. "
                    "Sertakan emoticon yang menarik dan sarankan 1-3 makanan spesifik. Jangan berikan pilihan mode lagi."
                )
                
                # Perbarui/sisipkan System Prompt di awal riwayat
                if messages_for_api and messages_for_api[0]["role"] == "system":
                    messages_for_api[0]["content"] = system_prompt_with_context
                else:
                    messages_for_api.insert(0, {"role": "system", "content": system_prompt_with_context})

                ai_response = get_ai_response(messages_for_api, "x-ai/grok-4-fast:free") 
                
                if ai_response:
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
                else:
                    st.session_state.messages.pop() # Hapus pesan user terakhir jika ada error

# Jika mode belum dipilih, tampilkan pesan selamat datang
elif st.session_state.mode is None:
    st.info("Silakan pilih mode **Masak** atau **Beli** untuk memulai generator.")