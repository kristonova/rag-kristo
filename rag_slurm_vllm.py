import os
from langchain_community.document_loaders import TextLoader, SitemapLoader
from bs4 import SoupStrainer
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

    # 1. Load dokumen dari sitemap wiki
    print("[1] Membaca semua halaman wiki dari sitemap...")
    loader = SitemapLoader(
        web_path="https://wiki.efisonlt.com/sitemap/sitemap-wiki.efisonlt.com-NS_0-0.xml",
        filter_urls=["https://wiki.efisonlt.com/wiki/"],
        requests_per_second=2,
        bs_kwargs={
            "parse_only": SoupStrainer("div", {"id": "mw-content-text"}),
        },
    )
    docs = loader.load()
    print(f"    → Total halaman dimuat: {len(docs)}")


    # 2. Potong Teks (Chunking)
    print("[2] Memotong teks menjadi bagian kecil...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4500, 
        chunk_overlap=300,
        separators=["\n---", "\n## ", "\n### ", "\n\n", "\n", " "],
    )  
    splits = text_splitter.split_documents(docs)


    # Tambahkan nama file sebagai metadata ke setiap chunk
    for s in splits:
        source = os.path.basename(s.metadata.get("source", ""))
        s.page_content = f"[Sumber: {source}]\n{s.page_content}"

    print(f"    → Jumlah chunk: {len(splits)}")

    # DEBUG: Tampilkan isi setiap chunk agar bisa dievaluasi
    for i, s in enumerate(splits):
        print(f"\n    [Chunk {i}] ({len(s.page_content)} chars):")
        print(f"    {s.page_content[:120]}...")


    # 3. Setup Model Embedding (Lokal via HuggingFace - CPU/GPU)
    print("[3] Load model embedding lokal...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

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
        max_new_tokens=1024,
        temperature=0.1,                           
        top_p=0.9,
        tensor_parallel_size=1,
        vllm_kwargs={
            "gpu_memory_utilization": 0.80,
            "enforce_eager": True,
            "max_model_len": 32768,
        }
    )


    # --- FASE 3: TANYA JAWAB (RETRIEVAL & GENERATION) ---

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # ...existing code...
    # Buat Prompt dengan format ChatML (untuk Qwen)
    template_qwen = """<|im_start|>system
Kamu adalah agen AI asisten admin HPC Slurm yang ahli. Tugasmu adalah membantu user berdasarkan dokumen referensi yang diberikan. Gunakan Bahasa Indonesia yang jelas.

Aturan:
1. Jawab HANYA berdasarkan dokumen referensi. KUTIP langkah-langkah dan perintah PERSIS seperti di dokumen. Jangan menambahkan langkah atau perintah yang tidak ada di dokumen.
2. Sertakan angka, nama, versi, dan spesifikasi PERSIS seperti tertulis di dokumen. Jangan membulatkan atau menambah presisi. Contoh: jika dokumen bilang ">=11", jawab ">=11", BUKAN "11.0" atau "11.2".
3. Jika informasi bisa DISIMPULKAN dari dokumen, berikan kesimpulan tersebut.
4. Jika informasi benar-benar TIDAK ADA di dokumen, katakan "Saya tidak menemukan informasi tersebut di sistem."
5. Jangan mengarang angka, rumus, perintah, URL, atau prosedur yang tidak ada di dokumen.
6. JANGAN mengganti perintah dari dokumen dengan perintah alternatif. Contoh: jika dokumen menulis "source activate", JANGAN ganti dengan "conda activate".
7. Bedakan "minimal" dan "maksimal". Jika dokumen hanya menyebutkan "minimal X" TANPA batas maksimal, jawab bahwa informasi batas maksimal tidak tersedia di dokumen.<im_end|>

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
        "Bagaimana cara membuat conda environment di aleleon?",
        "bagaimana cara menjalankan jupyter dengan conda environment sendiri?",
        "Versi Python default dari Anaconda3 2025.06-1 apa?",
        "Perintah apa untuk mengaktifkan Mamba 23.11.0-0?",
        "Bagaimana cara membuat modul pyload setelah conda env aktif?",
        "Perintah apa untuk melihat daftar modul pyload yang tersedia?",
        "Di partisi GPU mana batch job conda berjalan?",
        "Apa email support admin ALELEON?",
        "Jam kerja support EFISON kapan?",

        # ===== LEVEL 2: Gabungan Info (Multi-Chunk) =====
        "Apa saja pilihan cara menjalankan komputasi Python dengan conda env di ALELEON?",
        "Apa perbedaan antara menjalankan batch job via Job Composer EWS dan via terminal Slurm?",
        "Bagaimana langkah lengkap membuat conda env baru dan modul pyload dari awal?",
        "Apa saja status job di squeue dan artinya masing-masing?",
        "Bagaimana cara mengisi formulir Jupyter di EWS untuk conda env user?",

        # ===== LEVEL 3: Reasoning / Deduksi =====
        "Saya ingin pakai TensorFlow GPU di conda env. Package CUDA versi berapa yang harus saya instal?",
        "Kenapa Anaconda3 2024.06-1 tidak direkomendasikan? Apa yang harus dilakukan user yang sudah terpasang?",
        "Saya upload file Notebook (.ipynb) untuk batch job. Apa yang harus saya lakukan sebelum submit?",
        "Kenapa submit script menggunakan header #!/bin/bash -l dan perintah pyl load/pyl unload?",
        "Saya ingin menggunakan multi-GPU di ALELEON untuk deep learning. Package apa yang perlu diinstal?",
        "Storage HOME saya hampir penuh setelah banyak instal package conda. Bagaimana cara membersihkannya?",

        # ===== LEVEL 4: Anti-Hallucination (jawaban TIDAK ada di dokumen) =====
        "Berapa harga berlangganan conda env di ALELEON per bulan?",
        "Apakah ALELEON mendukung instalasi Docker di dalam conda env?",
        "Berapa jumlah maksimal GPU yang bisa diminta dalam satu batch job conda?",
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