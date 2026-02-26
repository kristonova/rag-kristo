# Penjelasan Lengkap Cara Kerja Kode RAG

## Arsitektur Keseluruhan

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────────┐ │
│  │ INGESTION│ → │EMBEDDING │ → │ RETRIEVAL │ → │ GENERATION  │ │
│  │ Pipeline │   │ + Store  │   │  (Search) │   │   (LLM)     │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────────┘ │
│                                                                 │
│  Fase 1: Data Masuk    Fase 2: Simpan Vektor   Fase 3: Jawab  │
└─────────────────────────────────────────────────────────────────┘
```

---

## FASE 1: Data Ingestion Pipeline

### Langkah 1 — Load Dokumen

```python
loader = TextLoader("spesifikasi_aleleon.txt")
docs = loader.load()
```

**Apa yang terjadi:**
- `TextLoader` membaca file `.txt` mentah menjadi objek `Document`
- Setiap `Document` punya 2 atribut:
  - `page_content`: isi teks
  - `metadata`: info file (nama, path)
- Pada tahap ini, **seluruh file = 1 dokumen besar**

```
spesifikasi_aleleon.txt (4076 chars)
        │
        ▼
┌──────────────────────────┐
│ Document(                │
│   page_content="Berikut  │
│   adalah konversi..."    │
│   metadata={source:...}  │
│ )                        │
└──────────────────────────┘
```

### Langkah 2 — Chunking (Pemotongan Teks)

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n---", "\n## ", "\n\n", "\n", " "],
)
splits = text_splitter.split_documents(docs)
```

**Apa yang terjadi:**

LLM punya batas context window. Kita tidak bisa memasukkan seluruh dokumen sekaligus. Jadi teks dipotong menjadi **chunk** kecil.

**Strategi `RecursiveCharacterTextSplitter`:**

Splitter ini mencoba memotong secara **hierarkis** — mulai dari separator terbesar, turun ke yang terkecil:

```
Prioritas potong:
  1. "\n---"   ← Potong di horizontal rule (pemisah seksi)
  2. "\n## "   ← Potong di heading Markdown level 2
  3. "\n\n"    ← Potong di paragraf kosong
  4. "\n"      ← Potong di baris baru
  5. " "       ← Potong di spasi (last resort)
```

**Kenapa "recursive"?** Karena jika potong di `\n---` menghasilkan chunk > 1000 chars, splitter **turun ke level berikutnya** (`\n## `), dan seterusnya sampai setiap chunk ≤ 1000 chars.

**Parameter:**

```
chunk_size=1000    → Maksimal 1000 karakter per chunk
chunk_overlap=200  → 200 karakter terakhir chunk N diulang di awal chunk N+1
```

**Kenapa overlap?** Agar informasi di perbatasan chunk tidak hilang:

```
Dokumen asli:
"...RAM per node: 500GB efektif | GPU per node: 2x RTX 3090..."

Tanpa overlap (chunk_overlap=0):
  Chunk 1: "...RAM per node: 500GB efek"  ← terpotong!
  Chunk 2: "tif | GPU per node: 2x RTX 3090..."

Dengan overlap (chunk_overlap=200):
  Chunk 1: "...RAM per node: 500GB efektif | GPU per no"
  Chunk 2: "500GB efektif | GPU per node: 2x RTX 3090..."
                ↑ overlap — info tidak hilang
```

**Hasil chunking dokumen Anda:**

```
Dokumen (4076 chars)
        │
        ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Chunk 0 │ │ Chunk 1 │ │ Chunk 2 │ │ Chunk 3 │ │ Chunk 4 │ │ Chunk 5 │ │ Chunk 6 │
│ 303 chr │ │ 778 chr │ │ 958 chr │ │  3 chr  │ │ 976 chr │ │ 907 chr │ │ 151 chr │
│ Intro   │ │ Compute │ │ Interac │ │  "---"  │ │Software │ │ PkgMgr  │ │ Footer  │
│ ALELEON │ │  Node   │ │  Node   │ │         │ │ Sistem  │ │ & Tools │ │         │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## FASE 2: Embedding + Vector Database

### Langkah 3 — Vector Embedding

```python
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

**Model yang dipakai: `all-MiniLM-L6-v2`**

| Properti | Detail |
|---|---|
| Arsitektur | MiniLM (distilled BERT) |
| Parameter | 22.7M |
| Dimensi output | **384 dimensi** |
| Max sequence | 256 tokens |
| Dilatih pada | 1 Billion sentence pairs |
| Jalan di | **CPU** (ringan, ~90MB) |

**Apa itu embedding?**

Embedding mengubah teks menjadi **vektor angka** di ruang 384 dimensi. Teks yang **mirip secara makna** akan punya vektor yang **dekat** satu sama lain.

```
"RAM per node di partisi epyc-jumbo adalah 500GB"
        │
        ▼  all-MiniLM-L6-v2
[0.032, -0.118, 0.245, ..., 0.067]    ← 384 angka

"Berapa jumlah RAM per node di partisi epyc-jumbo?"
        │
        ▼  all-MiniLM-L6-v2
[0.029, -0.121, 0.238, ..., 0.071]    ← 384 angka (MIRIP!)

"Apakah ALELEON mendukung Docker?"
        │
        ▼  all-MiniLM-L6-v2
[-0.156, 0.089, -0.034, ..., 0.193]   ← 384 angka (JAUH!)
```

**Ini BUKAN TF-IDF atau BM25.**

| Metode | Cara Kerja | Dipakai di kode ini? |
|---|---|---|
| **TF-IDF** | Hitung frekuensi kata. "RAM" muncul 3x = relevan. Tidak paham makna. | ❌ |
| **BM25** | TF-IDF yang lebih canggih dengan normalisasi panjang dokumen. | ❌ |
| **Sparse Retrieval** | Vektor besar tapi kebanyakan 0. Cocokkan kata kunci. | ❌ |
| **Dense Retrieval** ✅ | Teks → vektor padat 384D via neural network. Cocokkan **makna**. | ✅ **Ini yang dipakai** |

**Keunggulan Dense Retrieval:**

```
Query: "Saya butuh banyak memori untuk job saya"
  │
  ├── TF-IDF/BM25: Cari kata "memori" → TIDAK KETEMU (dokumen tulis "RAM")
  │
  └── Dense (MiniLM): Paham "memori" ≈ "RAM" secara semantik → KETEMU ✅
```

### Langkah 4 — Vector Database (Chroma)

```python
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
```

**Apa yang terjadi:**

1. Setiap chunk di-embed menjadi vektor 384D
2. Vektor + teks asli disimpan di database Chroma (in-memory)

```
Chroma DB (in-memory)
┌────────────────────────────────────────────────────┐
│ ID │ Vector (384D)              │ Teks Asli        │
├────┼────────────────────────────┼──────────────────┤
│ 0  │ [0.03, -0.12, 0.24, ...]  │ "Berikut adalah  │
│    │                            │  konversi..."    │
│ 1  │ [0.08, -0.05, 0.19, ...]  │ "## Compute Node │
│    │                            │  menjalankan..." │
│ 2  │ [-0.07, 0.14, 0.03, ...]  │ "## Interactive   │
│    │                            │  Node..."        │
│ ...│ ...                        │ ...              │
└────┴────────────────────────────┴──────────────────┘
```

**Chroma** adalah vector database yang:
- Ringan, berjalan **in-memory** (tidak perlu server terpisah)
- Mendukung **cosine similarity search**
- Cocok untuk prototyping (produksi biasanya pakai Pinecone, Weaviate, Milvus)

---

## FASE 3: Retrieval + Generation

### Retrieval — Cari Chunk Relevan

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

**Jenis retrieval: Approximate Nearest Neighbor (ANN) dengan Cosine Similarity**

Ketika user bertanya, proses yang terjadi:

```
User: "Berapa RAM di partisi epyc-jumbo?"
         │
         ▼ all-MiniLM-L6-v2
Query Vector: [0.029, -0.121, 0.238, ..., 0.071]
         │
         ▼ Cosine Similarity terhadap SEMUA chunk
         │
┌────────┬──────────────────────────────────┬────────────┐
│ Chunk  │ Isi                              │ Similarity │
├────────┼──────────────────────────────────┼────────────┤
│ 1      │ "Compute Node... RAM 500GB..."   │ 0.87 ← #1 │
│ 2      │ "Interactive Node... RAM 60GB.." │ 0.72 ← #2 │
│ 4      │ "Software Sistem..."             │ 0.41 ← #3 │
│ 0      │ "Intro ALELEON..."               │ 0.38 ← #4 │
│ 5      │ "Package Manager..."             │ 0.35 ← #5 │
│ 3      │ "---"                            │ 0.05       │
│ 6      │ "Footer..."                      │ 0.03       │
└────────┴──────────────────────────────────┴────────────┘
         │
         ▼ Ambil Top-K (k=5)
    Chunk 1, 2, 4, 0, 5 → dikirim ke LLM sebagai konteks
```

**Cosine Similarity Formula:**

```
                    A · B           Σ(Aᵢ × Bᵢ)
cos(θ) = ─────────────────── = ─────────────────────
              ||A|| × ||B||     √Σ(Aᵢ²) × √Σ(Bᵢ²)

Hasil: -1 (berlawanan) sampai +1 (identik)
```

### Prompt Template — Format ChatML

```python
template_qwen = """<|im_start|>system
Kamu adalah agen AI asisten admin HPC Slurm...
Jangan mengarang jawaban.<|im_end|>
<|im_start|>user
Dokumen Referensi:
{context}

Pertanyaan: {input}<|im_end|>
<|im_start|>assistant
"""
```

**Kenapa format `<|im_start|>` / `<|im_end|>`?**

Ini adalah **ChatML format** — format yang dipakai semasa training model Qwen untuk membedakan role:

```
<|im_start|>system     ← Instruksi untuk model (persona, rules)
...<|im_end|>
<|im_start|>user       ← Input dari user
...<|im_end|>
<|im_start|>assistant   ← Model mulai generate dari sini
```

**Template variables:**
- `{context}` → Diisi otomatis oleh LangChain dengan 5 chunk yang di-retrieve
- `{input}` → Diisi dengan pertanyaan user

### Generation — vLLM + Qwen2.5

```python
llm = VLLM(
    model="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_new_tokens=512,
    temperature=0.1,
    top_p=0.9,
    vllm_kwargs={
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "max_model_len": 4096,
    }
)
```

**Alur generation:**

```
┌─────────────────────────────────────────────────────────┐
│ Prompt yang dikirim ke LLM:                             │
│                                                         │
│ <|im_start|>system                                      │
│ Kamu adalah agen AI asisten admin HPC Slurm...          │
│ <|im_end|>                                              │
│ <|im_start|>user                                        │
│ Dokumen Referensi:                                      │
│ [Chunk 1] Compute Node... RAM epyc-jumbo 500GB...       │
│ [Chunk 2] Interactive Node... RAM 60GB...               │
│ [Chunk 4] Software Sistem... Rocky Linux 8...           │
│ [Chunk 0] Intro ALELEON Mk.V...                        │
│ [Chunk 5] Package Manager... EasyBuild...               │
│                                                         │
│ Pertanyaan: Berapa RAM di partisi epyc-jumbo?            │
│ <|im_end|>                                              │
│ <|im_start|>assistant                                   │
│                                                         │
│         ▼ Model generate token per token                │
│                                                         │
│ "Jumlah RAM per node di partisi epyc-jumbo              │
│  adalah 500GB efektif."                                 │
└─────────────────────────────────────────────────────────┘
```

**Parameter generation:**

| Parameter | Nilai | Arti |
|---|---|---|
| `temperature=0.1` | Sangat rendah → jawaban **deterministik**, tidak kreatif | Cocok untuk RAG (fakta) |
| `top_p=0.9` | Nucleus sampling — hanya pilih token dari 90% probabilitas tertinggi | Mengurangi jawaban random |
| `max_new_tokens=512` | Maksimal 512 token output | Batas panjang jawaban |
| `max_model_len=4096` | Maksimal 4096 token total (prompt + output) | Batas context window |
| `gpu_memory_utilization=0.90` | Pakai 90% VRAM | 10% sisakan untuk overhead |
| `enforce_eager=True` | Matikan CUDAGraph | Kompatibilitas RDNA4 |

### RAG Chain — Menggabungkan Semuanya

```python
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

**`create_stuff_documents_chain`** — Strategi: **"Stuff"**

"Stuff" artinya: **masukkan SEMUA chunk ke dalam 1 prompt sekaligus**.

```
Strategi lain (tidak dipakai di kode ini):
┌─────────────────────────────────────────────────────────┐
│ Stuff     : Semua chunk → 1 prompt → 1 jawaban    ✅   │
│ Map-Reduce: Tiap chunk → jawaban → gabung semua        │
│ Refine    : Chunk 1 → jawaban → + Chunk 2 → refine     │
│ Map-Rerank: Tiap chunk → jawaban + skor → pilih terbaik│
└─────────────────────────────────────────────────────────┘
```

**`create_retrieval_chain`** menggabungkan retriever + stuff chain:

```
User Input
    │
    ▼
┌──────────┐     ┌──────────────┐     ┌─────────┐
│ Retriever│ ──→ │ Stuff Chain  │ ──→ │  Output  │
│ (Top-5)  │     │ (Prompt+LLM) │     │ {answer} │
└──────────┘     └──────────────┘     └─────────┘
    │                    │
    │ 5 chunks           │ Prompt dengan
    │ relevan            │ context + question
    ▼                    ▼
 Dari Chroma        Ke vLLM/GPU
```

---

## Diagram Lengkap End-to-End

```
spesifikasi_aleleon.txt
        │
   [1] TextLoader
        │
        ▼
  1 Document (4076 chars)
        │
   [2] RecursiveCharacterTextSplitter
       (chunk_size=1000, overlap=200)
        │
        ▼
  7 Chunks (303, 778, 958, 3, 976, 907, 151 chars)
        │
   [3] all-MiniLM-L6-v2 (CPU, 22.7M params)
       Setiap chunk → vektor 384 dimensi
        │
        ▼
   [4] Chroma DB (in-memory)
       7 vektor + 7 teks tersimpan
        │
        │
   [5] vLLM + Qwen2.5-Coder-7B (GPU, 7B params)
       Model di-load ke VRAM (14.37 GiB)
        │
        │
  ══════╪══════════════════════════════════════
  Per pertanyaan:
        │
  User: "Berapa RAM epyc-jumbo?"
        │
        ▼
  [a] Embed pertanyaan → vektor 384D (CPU)
        │
  [b] Cosine similarity vs 7 chunk di Chroma
        │
  [c] Ambil top-5 chunk paling relevan
        │
  [d] Masukkan ke prompt template (ChatML)
        │
  [e] Kirim prompt ke Qwen2.5 via vLLM (GPU)
        │
  [f] Model generate jawaban token-by-token
        │
        ▼
  "Jumlah RAM per node di partisi
   epyc-jumbo adalah 500GB efektif."
```