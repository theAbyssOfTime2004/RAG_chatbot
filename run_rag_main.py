import torch
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
import os
import logging

logging.getLogger("transformers").setLevel(logging.WARNING)
# --- Constants and Configuration ---
PDF_FILE_PATH = "tóm tắt bài báo 2.pdf" 
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"
LLM_MODEL_NAME = "lmsys/vicuna-7b-v1.5"

def load_documents(file_path):
    """Loads documents from a PDF file."""
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy tệp PDF tại đường dẫn: {file_path}")
        return None
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Đã tải {len(documents)} trang từ tệp PDF.")
    return documents

def get_embeddings(model_name):
    """Initializes and returns HuggingFace embeddings."""
    print(f"Đang tải embedding model: {model_name}")
    return HuggingFaceEmbeddings(model_name=model_name)

def split_documents(documents, embeddings_model):
    """Splits documents using SemanticChunker."""
    if not documents:
        return []
    print("Đang chia nhỏ tài liệu bằng SemanticChunker...")
    semantic_splitter = SemanticChunker(
        embeddings=embeddings_model,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500, 
        add_start_index=True
    )
    docs = semantic_splitter.split_documents(documents)
    print(f"Số lượng chunks sau khi chia: {len(docs)}")
    return docs

def create_vector_store_and_retriever(docs, embeddings_model):
    """Creates a Chroma vector store and returns a retriever."""
    if not docs:
        print("Không có chunks nào để tạo vector store.")
        return None
    print("Đang tạo vector store Chroma...")
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings_model)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # Lấy top 3 chunks
    print("Đã tạo retriever.")
    return retriever

def load_llm_pipeline(model_name):
    """Loads the LLM with 4-bit quantization."""
    print(f"Đang tải LLM: {model_name} với lượng tử hóa 4-bit...")
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True,
        # device_map="auto" # Sẽ được xử lý bởi pipeline
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto" # Đảm bảo sử dụng GPU nếu có
    )
    llm = HuggingFacePipeline(pipeline=model_pipeline)
    print("LLM đã sẵn sàng.")
    return llm

def format_docs(docs_list):
  """Formats retrieved documents for the prompt."""
  return "\n\n".join(doc.page_content for doc in docs_list)

def build_rag_chain(retriever, llm):
    """Builds the RAG chain."""
    if not retriever or not llm:
        return None
    print("Đang xây dựng RAG chain...")
    #prompt_hub = hub.pull("rlm/rag-prompt")
    custom_prompt_template = """Bạn là một trợ lý AI hữu ích. Chỉ sử dụng thông tin được cung cấp trong ngữ cảnh sau để trả lời câu hỏi.
Nếu bạn không biết câu trả lời dựa trên ngữ cảnh, hãy nói rằng bạn không biết. Đừng cố bịa ra câu trả lời.
Hãy trả lời câu hỏi bằng tiếng Việt.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Trả lời (bằng tiếng Việt):"""
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt_template
        | llm
        | StrOutputParser()
    )
    print("RAG chain đã được xây dựng.")
    return rag_chain

def main():
    """Main function to run the RAG pipeline."""
    # 1. Load Documents
    documents = load_documents(PDF_FILE_PATH)
    if not documents:
        return

    # 2. Initialize Embeddings
    embedding_model = get_embeddings(EMBEDDING_MODEL_NAME)

    # 3. Split Documents
    split_docs = split_documents(documents, embedding_model)
    if not split_docs:
        return

    # 4. Create Vector Store and Retriever
    retriever = create_vector_store_and_retriever(split_docs, embedding_model)
    if not retriever:
        return

    # --- Kiểm tra retriever (tùy chọn) ---
    # test_query = "Phương Pháp Nghiên Cứu"
    # print(f"\nKiểm tra retriever với câu hỏi: '{test_query}'")
    # relevant_docs = retriever.invoke(test_query)
    # print(f"Số lượng tài liệu liên quan tìm thấy: {len(relevant_docs)}")
    # for i, doc_item in enumerate(relevant_docs):
    #     print(f"--- Tài liệu {i+1} ---")
    #     print(doc_item.page_content[:200] + "...") # In 200 ký tự đầu
    # print("-" * 20)
    # --- Kết thúc kiểm tra retriever ---

    # 5. Load LLM
    llm = load_llm_pipeline(LLM_MODEL_NAME)

    # 6. Build RAG Chain
    rag_chain = build_rag_chain(retriever, llm)
    if not rag_chain:
        return

    # 7. Ask a question
    user_question = input("Mời bạn nhập câu hỏi: ") 

    if not user_question:
        print("Không có câu hỏi nào được nhập. Kết thúc chương trình.")
        return

    print(f"\nCâu hỏi của bạn: {user_question}")
    print("Đang xử lý, vui lòng đợi...") # <--- THÊM DÒNG NÀY

    # Invoke chain và xử lý output
    try:
        output = rag_chain.invoke(user_question)
        # Xử lý output để lấy câu trả lời, có thể khác nhau tùy theo prompt
        if isinstance(output, str):
            if 'Answer:' in output:
                answer = output.split('Answer:')[1].strip()
            elif 'Trả lời:' in output: # Thêm trường hợp nếu prompt trả về tiếng Việt
                answer = output.split('Trả lời:')[1].strip()
            else:
                answer = output.strip() # Mặc định lấy toàn bộ output nếu không có marker
        else:
            answer = str(output) # Chuyển sang string nếu không phải

        print("\nTrả lời:")
        print(answer)
    except Exception as e:
        print(f"Đã xảy ra lỗi khi thực thi RAG chain: {e}")

if __name__ == "__main__":
    main()
