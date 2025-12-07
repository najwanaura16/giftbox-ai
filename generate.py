from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

model_name = "google/flan-t5-base"
lora_path = "giftbox-lora"

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, lora_path)

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs,
        max_length=250,
        temperature=0.9,
        top_p=0.9
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Coba generate
prompt = """
Jenis acara: Graduation.
Tema/mood: Friendly.
Warna tema: Biru.
Panjang teks: Sedang.
Jenis output: Caption Instagram.
Instruksi pengguna: Buat caption lucu buat temen yang baru lulus.
"""

print(generate_text(prompt))