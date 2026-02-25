import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import VLLM
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate


def main():
    print("Memulai proses RAG dengan mesin vLLM...\n")

    # --- FASE 1: MEMASUKKAN DATA (INGESTION) ---

    # 1. Load Dokumen
    print("[1] Membaca dokumen spesifikasi_aleleon.txt...")
    loader = TextLoader("spesifikasi_aleleon.txt")
    docs = loader.load()

    # 2. Potong Teks (Chunking)
    print("[2] Memotong teks menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n---", "\n## ", "\n\n", "\n", " "], # Prioritas potong di heading/separator)
    )  
    splits = text_splitter.split_documents(docs)
    print(f"    → Jumlah chunk: {len(splits)}")

    # DEBUG: Tampilkan isi setiap chunk agar bisa dievaluasi
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")


    # 3. Setup Model Embedding (Lokal via HuggingFace - CPU/GPU)
    print("[3] Load model embedding lokal...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Simpan ke Vector Database (Chroma)
    print("[4] Menyimpan vektor ke database Chroma...")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


    # --- FASE 2: SETUP vLLM (ENGINE INFERENCE) ---

    print("\n[5] Memuat model Qwen ke GPU menggunakan vLLM...")
    print("    (Ini akan memakan waktu untuk alokasi KV Cache di VRAM)")

    # Konfigurasi vLLM
    llm = VLLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        trust_remote_code=True,
        max_new_tokens=512,
        temperature=0.5,
        top_p=0.9,
        stop=["<|eot_id|>"],             # ← Stop string (bukan token ID)
        tensor_parallel_size=1,
        vllm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "max_model_len": 4096,
        }
    )


    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Buat Prompt dengan format LLAMA 3.1 (bukan ChatML!)
    template_llama = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user melakukan troubleshooting berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas dan mudah dipahami.
Jika jawabannya tidak ada di dokumen referensi, katakan "Saya tidak menemukan informasi tersebut di sistem." Jangan mengarang jawaban.<|eot_id|><|start_header_id|>user<|end_header_id|>

Dokumen Referensi:
{context}

Pertanyaan: {input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    prompt = PromptTemplate(
        template=template_llama,
        input_variables=["context", "input"]
    )

    # Rangkai rantai RAG (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- UJI COBA: BATCH SEMUA PERTANYAAN ---
    pertanyaan_list = [
        # LEVEL 1: Fakta Langsung
        "Berapa jumlah RAM per node di partisi epyc-jumbo?",
        "GPU apa yang digunakan di partisi ampere?",
        "Sistem operasi apa yang digunakan ALELEON Supercomputer?",

        # LEVEL 2: Gabungan Info
        "Apa perbedaan spesifikasi antara partisi epyc dan partisi ampere?",
        "Partisi mana saja yang memiliki GPU dan apa spesifikasi GPU di masing-masing partisi?",
        "Saya ingin menjalankan deep learning dengan GPU. Partisi mana yang harus saya gunakan dan berapa VRAM yang tersedia?",

        # LEVEL 3: Reasoning
        "Saya punya job yang butuh 400GB RAM. Partisi mana yang bisa saya gunakan?",
        "Apakah ALELEON mendukung container Docker? Jika tidak, alternatifnya apa?",
        "Saya ingin pakai Python 2 di ALELEON. Apakah bisa?",

        # LEVEL 4: Anti-Hallucination (jawaban TIDAK ada di dokumen)
        "Berapa biaya sewa per jam untuk menggunakan ALELEON Supercomputer?",
        "Apakah ALELEON mendukung GPU AMD Instinct MI300X?",
    ]

    # Batch invoke: kumpulkan semua input sekaligus lalu proses satu loop
    inputs = [{"input": q} for q in pertanyaan_list]

    for i, inp in enumerate(inputs, 1):
        print(f"\n{'='*60}")
        print(f"[Q{i}/{len(inputs)}] {inp['input']}")
        print("-" * 60)
        hasil = rag_chain.invoke(inp)
        print(hasil['answer'].strip())

    print(f"\n{'='*60}")
    print(f"Selesai — {len(inputs)} pertanyaan dijawab.")


# ============================================================
# INI KUNCINYA: Mencegah spawn menjalankan ulang seluruh script
# ============================================================
if __name__ == '__main__':
    main()