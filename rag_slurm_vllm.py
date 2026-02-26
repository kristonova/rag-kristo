import os
import glob
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

    # 1. Load SEMUA Dokumen .txt dari folder wiki/
    wiki_files = sorted(glob.glob("wiki/*.txt"))
    print(f"[1] Membaca {len(wiki_files)} dokumen dari folder wiki/...")
    docs = []
    for f in wiki_files:
        print(f"    → {f}")
        loader = TextLoader(f)
        docs.extend(loader.load())

        # ...existing code...
    
        # 2. Potong Teks (Chunking)
        print("[2] Memotong teks menjadi bagian kecil...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500, 
            chunk_overlap=300,
            separators=["\n---", "\n## ", "\n### ", "\n\n", "\n", " "],
        )  
        splits = text_splitter.split_documents(docs)
    
        # Tambahkan nama file sebagai metadata ke setiap chunk
        for s in splits:
            source = os.path.basename(s.metadata.get("source", ""))
            s.page_content = f"[Sumber: {source}]\n{s.page_content}"
    
        print(f"    → Jumlah chunk: {len(splits)}")
    
    # ...existing code...

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
        model="Qwen/Qwen2.5-Coder-7B-Instruct",  # ← Ganti kembali
        trust_remote_code=True,
        max_new_tokens=512,
        temperature=0.1,                           # ← Turunkan dari 0.6
        top_p=0.9,
        tensor_parallel_size=1,
        vllm_kwargs={
            "gpu_memory_utilization": 0.90,
            "enforce_eager": True,
            "max_model_len": 8192,
        }
    )


    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    # ...existing code...
    
        # Buat Prompt dengan format ChatML (untuk Qwen)
    template_qwen = """<|im_start|>system
    Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.
    
    Aturan:
    1. Jawab HANYA berdasarkan dokumen referensi. Sertakan angka, nama model, dan spesifikasi teknis yang spesifik jika ada di dokumen.
    2. Jika informasi bisa DISIMPULKAN dari dokumen (misal: hanya Python 3 yang tersedia berarti Python 2 tidak bisa), berikan kesimpulan tersebut.
    3. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
    4. Jangan mengarang angka, rumus, atau fakta yang tidak ada di dokumen.<|im_end|>
    <|im_start|>user
    Dokumen Referensi:
    {context}
    
    Pertanyaan: {input}<|im_end|>
    <|im_start|>assistant
    """
    
    prompt = PromptTemplate(
            template=template_qwen,
            input_variables=["context", "input"]
        )
    
    # ...existing code...

    # Rangkai rantai RAG (Chain)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- UJI COBA: BATCH SEMUA PERTANYAAN ---
    pertanyaan_list = [
        # ===== LEVEL 1: Fakta Langsung =====
        # [spesifikasi_aleleon.txt]
        "Berapa jumlah RAM per node di partisi epyc-jumbo?",
        "GPU apa yang digunakan di partisi ampere?",
        "Sistem operasi apa yang digunakan ALELEON Supercomputer?",
        # [mpi_aleleon_superkomputer.txt]
        "Implementasi MPI apa yang digunakan ALELEON Supercomputer?",
        # [tutorial_akun_trial_a6.txt]
        "Berapa limit maksimal concurrent job untuk akun uji coba trial A6?",
        # [komputasi_python_venv_user.txt]
        "Di direktori mana sebaiknya virtual environment Python ditempatkan di ALELEON?",

        # ===== LEVEL 2: Gabungan Info (Multi-Chunk) =====
        # [spesifikasi_aleleon.txt]
        "Apa perbedaan spesifikasi antara partisi epyc dan partisi ampere?",
        "Partisi mana saja yang memiliki GPU dan apa spesifikasi GPU di masing-masing partisi?",
        # [metode_komputasi_efison.txt]
        "Apa perbedaan antara batch job dan sesi interaktif di ALELEON?",
        # [tutorial_akun_trial_a6.txt]
        "Bagaimana cara login ke ALELEON Supercomputer? Sebutkan opsi yang tersedia.",
        # [mpi_aleleon_superkomputer.txt]
        "Apa perbedaan antara Pure MPI dan Hybrid MPI/OpenMP di ALELEON?",

        # ===== LEVEL 3: Reasoning / Deduksi =====
        # [spesifikasi_aleleon.txt]
        "Saya punya job yang butuh 400GB RAM. Partisi mana yang bisa saya gunakan?",
        "Apakah ALELEON mendukung container Docker? Jika tidak, alternatifnya apa?",
        # [komputasi_python_venv_user.txt + metode_komputasi_efison.txt]
        "Saya ingin pakai Python 2 di ALELEON. Apakah bisa?",
        # [tutorial_akun_trial_a6.txt]
        "Saya ingin menjalankan PyTorch di sesi Jupyter dengan GPU. Partisi apa yang harus saya pilih dan langkah apa saja yang perlu dilakukan?",
        # [mpi_aleleon_superkomputer.txt]
        "Saya ingin menjalankan 192 proses MPI dan butuh total 400GB RAM. Bagaimana cara mengisi SBATCH mem?",
        # [komputasi_python_venv_user.txt]
        "Kenapa saya harus mengaktifkan module python dan venv di dalam submit script SLURM, bukan hanya di terminal?",

        # ===== LEVEL 4: Anti-Hallucination (jawaban TIDAK ada di dokumen) =====
        "Berapa biaya sewa per jam untuk menggunakan ALELEON Supercomputer?",
        "Apakah ALELEON mendukung GPU AMD Instinct MI300X?",
        "Apakah ALELEON menyediakan akses ke cloud storage seperti AWS S3?",
        "Berapa kecepatan clock boost maksimal CPU EPYC 7702P di ALELEON?",
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