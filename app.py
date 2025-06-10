import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

# Khởi tạo session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embeddings_model():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm_model():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto" # Quan trọng cho việc phân bổ trên GPU/CPU
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
        # device_map="auto" không cần thiết ở đây nếu model đã được map
    )
    return HuggingFacePipeline(pipeline=model_pipeline)

def process_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()

    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings, # Sử dụng embeddings đã load
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)
    # Sửa lỗi cú pháp ở đây
    vector_db = Chroma.from_documents(documents=docs, embedding=st.session_state.embeddings)
    retriever = vector_db.as_retriever()
    prompt_hub = hub.pull("rlm/rag-prompt")

    def format_docs(docs_list):
        return "\n\n".join(doc.page_content for doc in docs_list)

    rag_chain_pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_hub
        | st.session_state.llm # Sử dụng LLM đã load
        | StrOutputParser()
    )
    
    os.unlink(tmp_file_path)
    return rag_chain_pipeline, len(docs)

# Giao diện Streamlit
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")

st.markdown("""
**Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
**Cách sử dụng đơn giản:**
1. **Upload PDF:** Chọn file PDF từ máy tính và nhấn "Xử lý PDF"
2. **Đặt câu hỏi:** Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức
---
""")

if not st.session_state.models_loaded:
    with st.spinner("Đang tải models... (có thể mất vài phút)"):
        st.session_state.embeddings = load_embeddings_model()
        st.session_state.llm = load_llm_model()
        st.session_state.models_loaded = True
    st.success("Models đã sẵn sàng!")
    st.rerun() # Rerun để giao diện cập nhật sau khi model load xong

uploaded_file = st.file_uploader("Upload file PDF", type="pdf")

if uploaded_file:
    if st.button("Xử lý PDF"):
        with st.spinner("Đang xử lý PDF..."):
            st.session_state.rag_chain, num_chunks = process_pdf_documents(uploaded_file)
            st.success(f"Hoàn thành! Tài liệu được chia thành {num_chunks} chunks.")

if st.session_state.rag_chain:
    question = st.text_input("Đặt câu hỏi về nội dung PDF:")
    if question:
        with st.spinner("Đang tìm câu trả lời..."):
            output = st.session_state.rag_chain.invoke(question)
            # Sửa lỗi thụt lề và xử lý output
            if "Answer:" in output:
                answer = output.split("Answer:")[1].strip()
            else:
                answer = output.strip()
            
            st.write("**Trả lời:**")
            st.write(answer)
