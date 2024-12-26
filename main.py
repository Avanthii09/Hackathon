import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import tempfile
import fitz  
import docx 
import os  

API_KEY = "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"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        return file.read()

def extract_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        file_path = tmp_file.name

    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        return extract_text_from_pdf(file_path)
    elif file_extension == "docx":
        return extract_text_from_docx(file_path)
    elif file_extension == "txt":
        return extract_text_from_txt(file_path)
    else:
        st.error("Unsupported file type. Please upload a .pdf, .docx, or .txt file.")
        return None

def split_text_into_chunks(text, chunk_size=1000):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

def embed_chunks(chunks):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.cpu().numpy())
    return index

def retrieve_context(query, index, chunks, model, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [chunks[i] for i in indices[0]]

def chatbot_context_retrieval(uploaded_file, query):
    text = extract_text(uploaded_file)
    if not text:
        return ["Error extracting text from the file."]

    chunks = split_text_into_chunks(text, chunk_size=1000)
    embeddings = embed_chunks(chunks)
    faiss_index = create_faiss_index(embeddings)
    return retrieve_context(query, faiss_index, chunks, SentenceTransformer('paraphrase-MiniLM-L6-v2'))



def main():
    st.set_page_config(page_title="Doc Chat", layout="wide")

    st.markdown("""
    <style>
 
    .uploadedFile {display: none;}
    .stFileUploader > div:first-child {margin-top: 0;}

   
    .main > div {
        max-width: 1000px;
        margin: 5rem auto;
        padding: 0;
    }
    
    .chat-header {
        text-align: center;
        margin: 1rem 0;
        color: #ffffff;
        font-size: 2rem;
        font-weight: 600;
    }

  
    .chat-messages {
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 1rem;
    }

    .message {
        margin: 0.5rem 0;
        padding: 1rem;
        border-radius: 4px;
        max-width: 80%;
    }

    .user-message {
        background-color: #e6f0ff;
        color: #333;
        margin-left: auto;
    }

    .bot-message {
        background-color: #f0f0f0;
        color: #333;
        margin-right: auto;
    }

    .stTextInput > div > div {
        border-radius: 20px;
        padding: 0.7rem 1rem;
    }

    .stButton > button {
        border-radius: 20px;
        padding: 0.7rem 1.5rem;
    }

    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<div class='chat-header'>💬 Doc Chat</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if uploaded_file:
            file_info = f"**File Name:** {uploaded_file.name}  \n**File Size:** {uploaded_file.size / 1024:.2f} KB"
            st.info(file_info)

            with st.container():
                st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
                for msg in st.session_state["messages"]:
                    if msg["role"] == "user":
                        st.markdown(
                            f'<div class="message user-message">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="message bot-message">{msg["content"]}</div>',
                            unsafe_allow_html=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)

                query = st.text_input("", placeholder="Type your message here...", key="query_input")
                send_button = st.button("Send", key="send")

                if send_button and query:
                    st.session_state["messages"].append({"role": "user", "content": query})
                    context = chatbot_context_retrieval(uploaded_file, query)
                    context_text = "\n".join(context)

                    client = Groq(api_key=API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": context_text + "\n" + query}],
                        model="llama3-70b-8192",
                    )
                    response = chat_completion.choices[0].message.content
                    st.session_state["messages"].append({"role": "bot", "content": response})
                    st.experimental_rerun()

if __name__ == "__main__":
    main()
