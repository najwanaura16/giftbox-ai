import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="GiftBox AI", page_icon="🎁")
st.title("🎁 GiftBox AI – Generator Teks Ucapan")

model_name = "google/flan-t5-base"   # kecil tapi jauuuh lebih bagus

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# UI
jenis_acara = st.selectbox("Jenis Acara", [
    "Ulang Tahun", "Wedding", "Anniversary", "Graduation",
    "Hari Ibu / Ayah", "Lainnya"
])

tema = st.selectbox("Tema/Mood", [
    "Romantis", "Elegan", "Puitis", "Lucu", "Formal",
    "Friendly", "Minimalis"
])

warna = st.text_input("Warna Tema (opsional)")
panjang = st.selectbox("Panjang Teks", ["Pendek", "Sedang", "Panjang"])
jenis_output = st.selectbox("Jenis Output", [
    "Ucapan Singkat", "Cerita Mini", "Kartu Ucapan Formal",
    "Caption Instagram", "Pesan Personal"
])
instruksi = st.text_area("Instruksi Tambahan (opsional)")

if st.button("Generate"):
    prompt = (
        f"Buatkan {jenis_output.lower()} untuk {jenis_acara.lower()} "
        f"dengan gaya {tema.lower()}. "
        f"Panjang teks: {panjang.lower()}. "
        f"Warna tema: {warna}. "
        f"Instruksi tambahan: {instruksi}. "
        f"Tulis dengan bahasa Indonesia yang natural dan menarik."
    )

    with st.spinner("✨ Menghasilkan teks..."):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        output = model.generate(
            **inputs,
            max_length=220,
            num_beams=4,
            temperature=0.9,
            top_p=0.95,
            do_sample=True
        )

        result = tokenizer.decode(output[0], skip_special_tokens=True)

        st.success(result)
