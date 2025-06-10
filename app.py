import streamlit as st
import tempfile
import os
import torch # Cần thiết cho BitsAndBytesConfig
import logging

from langchain_community.document_loaders import UnstructuredPDFLoader # MỤC 1: Cải thiện PDF Loader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate # MỤC 5: Prompt Engineering
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores.utils import filter_complex_metadata

# MỤC 3: Tối ưu hóa Retrieval (Re-ranking)
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from sentence_transformers.cross_encoder import CrossEncoder


# Giảm output không cần thiết từ các thư viện
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("unstructured").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


# Khởi tạo session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "reranker" not in st.session_state: # Thêm reranker vào session state
    st.session_state.reranker = None


@st.cache_resource
def load_embeddings_model():
    # MỤC 2: Nâng cấp Mô hình Embedding cho tiếng Anh và thuật ngữ khoa học
    # MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder" # Mô hình cũ
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" # Mô hình mới, mạnh cho tiếng Anh
    st.write(f"Đang tải Embedding Model: {MODEL_NAME}")
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)

@st.cache_resource
def load_llm_model():
    MODEL_NAME = "lmsys/vicuna-7b-v1.5" # Giữ nguyên LLM hoặc có thể nâng cấp nếu có tài nguyên
    st.write(f"Đang tải LLM: {MODEL_NAME}")
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
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024, # Tăng max_new_tokens cho các câu trả lời chi tiết hơn
        pad_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=model_pipeline)

@st.cache_resource
def load_reranker_model():
    # MỤC 3: Tải mô hình CrossEncoder cho Re-ranking
    RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    st.write(f"Đang tải Re-ranker Model: {RERANKER_MODEL_NAME}")
    return CrossEncoder(RERANKER_MODEL_NAME)

def process_pdf_documents(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # MỤC 1: Sử dụng UnstructuredPDFLoader
    st.write("Đang phân tích PDF với UnstructuredPDFLoader...")
    loader = UnstructuredPDFLoader(tmp_file_path, mode="elements", strategy="fast") 
    documents = loader.load()

    st.write("Đang chia nhỏ tài liệu với SemanticChunker...")
    semantic_splitter = SemanticChunker(
        embeddings=st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=256, 
        add_start_index=True
    )
    docs = semantic_splitter.split_documents(documents)
    
    # LỌC METADATA PHỨC TẠP TRƯỚC KHI ĐƯA VÀO CHROMA
    st.write("Đang lọc metadata phức tạp...")
    filtered_docs = filter_complex_metadata(docs)
    
    st.write("Đang tạo Vector Store (ChromaDB)...")
    # SỬ DỤNG filtered_docs THAY VÌ docs
    vector_db = Chroma.from_documents(documents=filtered_docs, embedding=st.session_state.embeddings)
    
    # MỤC 3: Tối ưu hóa Retrieval với Re-ranking
    st.write("Đang cấu hình Retriever với Re-ranker...")
    base_retriever = vector_db.as_retriever(search_kwargs={"k": 10}) # Lấy nhiều docs hơn cho re-ranker
    
    compressor = CrossEncoderReranker(model=st.session_state.reranker, top_n=3) # Lấy top 3 sau khi re-rank
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    retriever = compression_retriever # Sử dụng retriever đã được nén và re-rank

    # MỤC 5: Prompt Engineering Nâng cao cho Scientific Papers (Tiếng Anh)
    # Do embedding model đã chuyển sang tiếng Anh, prompt cũng nên hướng dẫn LLM trả lời bằng tiếng Anh.
    # Nếu bạn muốn tiếng Việt, cần đảm bảo LLM và embedding model hỗ trợ tốt và prompt phải rất rõ ràng.
    custom_prompt_template_en = """You are a highly intelligent AI assistant specialized in understanding and answering questions about scientific papers.
Use the following retrieved context from a scientific paper to answer the question.
If you don't know the answer based on the context, just say that you don't know. Do not try to make up an answer.
Be concise and factual. If possible, try to synthesize information from multiple parts of the context.
Answer the question in English.

Context:
{context}

Question: {question}

Answer (in English):"""
    
    prompt = ChatPromptTemplate.from_template(custom_prompt_template_en)

    def format_docs(docs_list):
        # Thêm nguồn (metadata) nếu có và hữu ích
        formatted_docs = []
        for i, doc in enumerate(docs_list):
            content = doc.page_content
            source_info = ""
            if doc.metadata:
                if 'page_number' in doc.metadata:
                    source_info += f" [Page: {doc.metadata['page_number']}]"
                # Bạn có thể thêm các metadata khác nếu UnstructuredPDFLoader cung cấp
            formatted_docs.append(f"Source Document [{i+1}]{source_info}:\n{content}")
        return "\n\n".join(formatted_docs)


    rag_chain_pipeline = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm 
        | StrOutputParser()
    )
    
    os.unlink(tmp_file_path) # Xóa tệp tạm
    return rag_chain_pipeline, len(docs)

# Giao diện Streamlit
st.set_page_config(page_title="Advanced PDF RAG Assistant", layout="wide")
st.title("Advanced PDF RAG Assistant")

st.markdown("""
**AI Assistant for Scientific Papers**
**How to use:**
1. **Upload PDF:** Select a PDF file (preferably a scientific paper in English for best results with current models) and click "Process PDF".
2. **Ask Question:** Type your question about the document content and get an answer.
---
""")

# Khởi tạo các session_state keys nếu chúng chưa tồn tại
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "reranker" not in st.session_state:
    st.session_state.reranker = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# Chỉ tải models nếu chúng chưa được tải
if not st.session_state.models_loaded:
    with st.spinner("Loading models... This may take a few minutes, especially on the first run."):
        models_successfully_loaded = False
        try:
            st.session_state.embeddings = load_embeddings_model()
            st.session_state.llm = load_llm_model()
            st.session_state.reranker = load_reranker_model() # Tải reranker model

            if st.session_state.embeddings and st.session_state.llm and st.session_state.reranker:
                st.session_state.models_loaded = True
                models_successfully_loaded = True
                st.success("Models are ready!")
            else:
                # Xác định model nào không tải được
                missing_models = []
                if not st.session_state.embeddings: missing_models.append("Embeddings")
                if not st.session_state.llm: missing_models.append("LLM")
                if not st.session_state.reranker: missing_models.append("Re-ranker")
                st.error(f"Failed to load the following models: {', '.join(missing_models)}. Please check the logs.")
        except Exception as e:
            st.error(f"An error occurred while loading models: {e}")
            st.exception(e) # In traceback đầy đủ vào log và giao diện

        if models_successfully_loaded:
            st.rerun() # Rerun chỉ khi tất cả model đã được tải thành công

# Nếu models_loaded vẫn là False sau khi cố gắng tải, hiển thị thông báo và dừng
if not st.session_state.models_loaded:
    st.error("Models could not be loaded. The application cannot proceed. Please check the console logs for errors and try refreshing the page.")
    st.stop() # Dừng thực thi script nếu model không tải được


uploaded_file = st.file_uploader("Upload your Scientific Paper (PDF)", type="pdf")

if uploaded_file:
    if st.button("Process PDF"):
        # Kiểm tra lại ở đây vì st.rerun có thể làm thay đổi luồng
        if not st.session_state.models_loaded or st.session_state.reranker is None:
            st.error("Models are not fully loaded or Re-ranker is not available. Please wait or refresh. If the problem persists, check logs.")
        else:
            with st.spinner("Processing PDF and building RAG chain... This might take a while."):
                try:
                    st.session_state.rag_chain, num_chunks = process_pdf_documents(uploaded_file)
                    st.success(f"Processing complete! Document divided into {num_chunks} initial chunks.")
                except Exception as e:
                    st.error(f"An error occurred during PDF processing: {e}")
                    st.exception(e)


if st.session_state.rag_chain:
    question = st.text_input("Ask a question about the PDF content (in English):")
    if question:
        with st.spinner("Thinking..."):
            try:
                output = st.session_state.rag_chain.invoke(question)
                answer = output.strip()
                st.write("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred while getting the answer: {e}")
                st.exception(e)

st.markdown("---")
st.markdown("Models used: `sentence-transformers/all-mpnet-base-v2` (Embeddings), `lmsys/vicuna-7b-v1.5` (LLM), `cross-encoder/ms-marco-MiniLM-L-6-v2` (Re-ranker).")