import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="GiftBox AI", page_icon="🎁")
st.title("🎁 GiftBox AI – Generator Teks Ucapan")

model_name = "google/flan-t5-small"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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

warna = st.text_input("Warna Tema")
panjang = st.selectbox("Panjang Teks", ["Pendek", "Sedang", "Panjang"])
jenis_output = st.selectbox("Jenis Output", [
    "Ucapan Singkat", "Cerita Mini", "Kartu Ucapan Formal",
    "Caption Instagram", "Pesan Personal"
])
instruksi = st.text_area("Instruksi Tambahan")

if st.button("Generate"):
    prompt = (
        f"Jenis acara: {jenis_acara}. Tema/mood: {tema}. Warna tema: {warna}. "
        f"Panjang teks: {panjang}. Jenis output: {jenis_output}. "
        f"Instruksi pengguna: {instruksi}"
    )
    with st.spinner("✨ Sedang menghasilkan teks..."):
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_length=180)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        st.success(result)
