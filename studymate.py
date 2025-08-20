import streamlit as st
import fitz  # PyMuPDF
import tempfile
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import pipeline
from huggingface_hub import login
import google.generativeai as genai

# 1️⃣ Hugging Face login
login(token="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")  # Replace with your actual Hugging Face token

# 2️⃣ Gemini setup
genai.configure(api_key="AIzaSyDAhbTL2yOb0juBy-f6Fl4nQ2nuNMo96_A")  # Replace with your Gemini API Key
MODEL_NAME = "models/gemini-1.5-flash"  # Correct model name format from ListModels

# 3️⃣ Hugging Face model for context-based answer
generator = pipeline("text2text-generation", model="google/flan-t5-base")

# 4️⃣ Sentence transformer for embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- PDF processing ---
def read_pdf(path):
    with fitz.open(path) as doc:
        return "".join(page.get_text() for page in doc)

def chunk_text(text, size=500):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def embed_chunks(chunks):
    return embedder.encode(chunks, convert_to_tensor=True)

def build_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.detach().cpu().numpy())
    return index

def search_chunks(index, query, chunks, embeddings, threshold=0.4):
    q = embedder.encode([query], convert_to_tensor=True)
    D, I = index.search(q.cpu().numpy(), 3)
    sims = [util.cos_sim(q, embeddings[i])[0][0].item() for i in I[0]]
    if max(sims) < threshold:
        return None
    return " ".join(chunks[i] for i in I[0])

# --- Answer Generation ---
def generate_answer(question, context=None):
    if context:
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        return generator(prompt, max_new_tokens=100)[0]["generated_text"]
    else:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(question)
        return response.text

# --- Streamlit App UI ---
st.set_page_config(page_title="📘 StudyMate Chatbot", layout="centered")
st.markdown("<h1 align='center'>📚 StudyMate: Chat with Your PDF or Ask Anything!</h1>", unsafe_allow_html=True)
st.markdown("---")

pdf = st.file_uploader("Upload your PDF", type="pdf")
if pdf:
    with st.spinner("Reading PDF..."):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf.read())
        tmp_path = tmp.name
    text = read_pdf(tmp_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)
    index = build_index(embeddings)

    st.markdown("### 💬 Ask your question:")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_input("You:", key="input")
    if question:
        context = search_chunks(index, question, chunks, embeddings)
        answer = generate_answer(question, context)

        st.session_state.chat_history.append(("🧑‍🎓 You", question))
        st.session_state.chat_history.append(("🤖 StudyMate", answer))

    for sender, msg in st.session_state.chat_history:
        with st.chat_message(sender):
            st.markdown(msg)

st.markdown("---")
st.markdown("<p align='center' style='font-size:12px;'>Powered by Hugging Face & Gemini · Built for CognitiveX</p>", unsafe_allow_html=True)