import os
import sys

# ==========================================================
# Alihkan Cache Hugging Face ke Drive D
# ==========================================================
# Secara default, Hugging Face menyimpan model di C:\Users\...\.cache
# Kode ini mengalihkannya ke folder 'hf_cache' di dalam project (Drive D)
project_dir = os.path.dirname(os.path.abspath(__file__))
hf_cache_dir = os.path.join(project_dir, "hf_cache")
os.makedirs(hf_cache_dir, exist_ok=True)

os.environ["HF_HOME"] = hf_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["HUGGINGFACE_HUB_CACHE"] = hf_cache_dir

import threading
from threading import Thread
import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# ==========================================================
# Konfigurasi Halaman Streamlit
# ==========================================================
st.set_page_config(
    page_title="Llama-2 Fine-Tuned Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 8px 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown('<div class="main-header">🦙 Llama-2 Mini-Guanaco Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Web interface untuk model Llama-2-7b yang telah di-fine-tune menggunakan QLoRA pada dataset Guanaco 1k.</div>',
    unsafe_allow_html=True,
)

# ==========================================================
# Sidebar: Pengaturan Parameter & Info Hardware
# ==========================================================
with st.sidebar:
    st.header("⚙️ Konfigurasi Model")
    
    device_info = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"Hardware: **{device_info}**\n\n{gpu_name}")
    else:
        st.warning(f"Hardware: **{device_info}**\n\n*Catatan: Tanpa GPU Nvidia (CUDA), loading model 7B akan membutuhkan RAM besar dan inferensi lebih lambat.*")

    st.subheader("Parameter Generasi")
    temperature = st.slider(
        "Temperature",
        min_value=0.01,
        max_value=1.5,
        value=0.7,
        step=0.05,
        help="Semakin tinggi nilai, respon akan semakin kreatif. Nilai lebih rendah membuat respon lebih terprediksi.",
    )
    top_p = st.slider(
        "Top-p (Nucleus Sampling)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Memilih token dari probabilitas kumulatif teratas.",
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        min_value=1.0,
        max_value=2.0,
        value=1.15,
        step=0.05,
        help="Mencegah model mengulang-ulang kalimat atau kata yang sama.",
    )
    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=64,
        max_value=1024,
        value=256,
        step=32,
        help="Jumlah token/kata maksimum yang dihasilkan dalam sekali respon.",
    )

    st.divider()

    # Tombol Reset Chat
    if st.button("🗑️ Bersihkan Riwayat Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya asisten Llama-2 hasil fine-tuning Anda. Ada yang bisa saya bantu hari ini?"}
        ]
        st.rerun()

    st.markdown("---")
    st.caption("📂 Bobot Adapter: `llama-2-7b-miniguanaco`\n\n🏛️ Base Model: `NousResearch/Llama-2-7b-hf`")

# ==========================================================
# Load Model & Tokenizer (Cached)
# ==========================================================
@st.cache_resource(show_spinner="Sedang memuat Base Model Llama-2 & LoRA Adapter ke memori...")
def load_llm_engine():
    base_model_name = "NousResearch/Llama-2-7b-hf"
    
    # Deteksi path adapter lokal (mendukung path relatif atau absolut)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(current_dir, "llama-2-7b-miniguanaco")
    
    if not os.path.exists(adapter_path):
        adapter_path = "./llama-2-7b-miniguanaco"

    has_cuda = torch.cuda.is_available()

    # Load Tokenizer dari adapter folder atau base model
    if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=hf_cache_dir, trust_remote_code=True)
        
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Konfigurasi Model
    if has_cuda:
        # 4-bit Quantization (QLoRA) persis seperti di Colab
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_dir,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    else:
        # Fallback untuk perangkat non-CUDA (CPU)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            cache_dir=hf_cache_dir,
            device_map="cpu",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    base_model.config.use_cache = True

    # Pasang adapter LoRA hasil fine-tuning
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer

# Muat model
try:
    model, tokenizer = load_llm_engine()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.info("Pastikan Anda memiliki koneksi internet untuk mengunduh base model pertama kali, dan library pendukung sudah terinstall.")
    st.stop()

# ==========================================================
# Manajemen Riwayat Chat
# ==========================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya asisten Llama-2 hasil fine-tuning Anda. Ada yang bisa saya bantu hari ini?"}
    ]

# Tampilkan chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================================
# Generator Streaming Inferensi
# ==========================================================
def stream_llm_response(user_prompt: str):
    # Format prompt persis seperti saat proses fine-tuning Guanaco
    system_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
    )
    b_inst, e_inst = "### Instruction:\n", "### Response:\n"
    formatted_prompt = f"{system_prompt}{b_inst}{user_prompt.strip()}\n\n{e_inst}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)

    # TextIteratorStreamer untuk efek mengetik real-time
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # Jalankan inference di thread terpisah agar UI Streamlit tidak blocking
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield token satu per satu ke Streamlit
    for new_text in streamer:
        yield new_text

# ==========================================================
# Input Pengguna
# ==========================================================
if user_prompt := st.chat_input("Ketik pertanyaan atau instruksi Anda di sini..."):
    # Tampilkan prompt pengguna di UI
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Hasilkan dan tampilkan respon asisten secara streaming
    with st.chat_message("assistant"):
        response = st.write_stream(stream_llm_response(user_prompt))

    # Simpan respon ke session history
    st.session_state.messages.append({"role": "assistant", "content": response})

