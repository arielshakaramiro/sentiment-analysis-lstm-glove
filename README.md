# Sentiment Analysis: Word Embedding (GloVe) + Bidirectional LSTM

Klasifikasi sentimen review film (IMDB, positif/negatif) menggunakan **pretrained word embedding (GloVe)** sebagai representasi kata dan **Bidirectional LSTM** sebagai encoder sekuensial. Dilengkapi notebook deployment untuk membungkus model jadi REST API.

## Arsitektur

`Embedding (diinisialisasi dari GloVe 100d) → Bidirectional LSTM (2 layer) → Dropout → Linear (binary output)`

> **Catatan akurasi:** bobot embedding diinisialisasi dari GloVe, tapi **tidak dibekukan (not frozen)** — layer embedding ikut ter-*fine-tune* selama training bersama layer lainnya, karena tidak ada baris kode yang men-set `requires_grad=False` pada layer ini.

| Komponen | Nilai |
|---|---|
| Vocabulary size | 25,002 |
| Embedding dimension | 100 (GloVe 6B.100d) |
| Hidden dimension | 256 |
| LSTM layers | 2, bidirectional |
| Dropout | 0.5 |
| Trainable parameters | 4,810,857 |
| Optimizer | Adam |
| Loss function | BCEWithLogitsLoss |

> **Catatan:** di source aslinya class model ini dinamai `RNN`, padahal implementasinya memakai `nn.LSTM`. Di repo ini sudah diganti jadi `SentimentLSTM` supaya tidak membingungkan.

## Hasil Training (Terverifikasi, 5 Epoch)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.658 | 60.43% | 0.540 | 72.92% |
| 2 | 0.556 | 71.12% | 0.440 | 80.07% |
| 3 | 0.417 | 81.63% | 0.341 | 85.54% |
| 4 | 0.321 | 87.10% | 0.327 | 86.75% |
| 5 | 0.284 | 88.73% | 0.295 | 88.00% |

**Test set:** Loss 0.303 | Accuracy **87.75%**

## Contoh Prediksi

| Kalimat | Skor (0=negatif, 1=positif) | Interpretasi |
|---|---|---|
| "This film is terrible" | 0.0055 | Sangat negatif ✅ |
| "This film is great" | 0.9846 | Sangat positif ✅ |

## Deployment (FastAPI + ngrok)

Notebook kedua (`sentiment_analysis_api_deployment.ipynb`) membungkus model jadi REST API:

- `GET /` — health check
- `POST /predict/` — input `{"sentence": "..."}`, output skor + label (`very positive` / `positive` / `neutral` / `negative`)

Contoh response nyata (model yang sama, dipanggil lewat API):
```json
{"sentence": "This film is great", "sentiment": "very positive", "score": 0.9846353530883789}
```

### ⚠️ Keamanan ngrok

Source asli notebook deployment sempat menyertakan **ngrok authtoken dalam bentuk plain text**. Di repo ini sudah diganti dengan pola aman: token diambil dari **Colab Secrets** (`google.colab.userdata`), bukan hardcode. Kalau menjalankan ulang, set secret `NGROK_AUTHTOKEN` di Colab kamu sendiri — jangan pernah commit token asli ke repo publik.

## Keterbatasan

- Model dilatih pada review film berbahasa Inggris — tidak akan bekerja baik untuk teks berbahasa lain tanpa fine-tuning ulang.
- Vocabulary dibatasi 25,000 kata paling sering muncul; kata di luar itu dipetakan ke token `<unk>`.
- Public URL ngrok bersifat sementara dan berubah setiap kali tunnel baru dibuka (kecuali pakai ngrok berbayar dengan custom domain).
- GloVe adalah representasi kata **statis** (satu kata = satu vektor tetap, terlepas dari konteks kalimat). Untuk kebutuhan produksi saat ini, model berbasis Transformer (BERT, RoBERTa, DistilBERT) umumnya memberi akurasi lebih tinggi karena representasinya kontekstual — kata yang sama bisa punya makna berbeda tergantung kalimatnya.

## Cara Menjalankan

1. Jalankan `sentiment_analysis_lstm_training.ipynb` di Google Colab (disarankan GPU) sampai selesai — ini akan menghasilkan `sentiment-model.pt` dan `vocab.txt`.
2. Jalankan `sentiment_analysis_api_deployment.ipynb` untuk memuat model tersebut dan mengekspos API lewat ngrok.
3. Set secret `NGROK_AUTHTOKEN` di Colab Secrets sebelum menjalankan cell ngrok.

## Dataset & Referensi

- Dataset: [IMDB Movie Reviews](https://ai.stanford.edu/~amaas/data/sentiment/) (via `torchtext.datasets`)
- Word embedding: [GloVe 6B.100d](https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt)

## Struktur Repo

```
.
├── sentiment_analysis_lstm_training.ipynb     # Training: data, vocab, GloVe, model, evaluasi
├── sentiment_analysis_api_deployment.ipynb    # Deployment: FastAPI + ngrok
└── README.md
```
