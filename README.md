# IMDB Sentiment Analysis | IMDB Duygu Analizi
### BoW/TF-IDF vs BiLSTM vs DistilBERT

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🇹🇷 Türkçe

### Proje Hakkında

Bu proje, IMDB platformundan derlenen 50.000 film yorumu üzerinde duygu analizi gerçekleştirmektedir. Üç farklı makine öğrenmesi yaklaşımı karşılaştırmalı olarak değerlendirilmiştir:

| Model | Accuracy | F1 Score | ROC-AUC | Eğitim Süresi |
|-------|----------|----------|---------|---------------|
| TF-IDF + Logistic Regression | %89.52 | 0.8951 | 0.9580 | ~2 dakika |
| TF-IDF + SVM | %90.10 | 0.9009 | 0.9601 | ~3 dakika |
| BiLSTM | %84.41 | 0.8433 | 0.9198 | 3.9 dakika |
| **DistilBERT** | **%91.32** | **0.9137** | **0.9724** | 20.3 dakika |

### Gereksinimler

- Windows 10/11, macOS veya Linux
- Python 3.10, 3.11 veya 3.12
- En az 8 GB RAM
- GPU (isteğe bağlı, ama önerilir) — NVIDIA GPU + CUDA

### Kurulum — Adım Adım

#### 1. Miniconda İndir ve Kur

Miniconda, Python ortamlarını yönetmek için kullanılan ücretsiz bir araçtır.

👉 https://docs.anaconda.com/miniconda/ adresinden **Windows 64-bit** sürümünü indir ve kur.

Kurulum sırasında:
- **"Just Me (recommended)"** seç
- **"Add to PATH"** kutusunu işaretle

#### 2. Projeyi İndir

**Seçenek A — Git ile (önerilir):**
```bash
git clone https://github.com/edipgenc/imdb-sentiment.git
cd imdb-sentiment
```

**Seçenek B — ZIP olarak:**
- Sayfanın sağ üstünden **Code → Download ZIP** tıkla
- ZIP'i bir klasöre çıkart
- O klasörü terminalde aç

#### 3. Conda Ortamını Oluştur

Başlat menüsünden **Anaconda Prompt** aç ve şu komutları çalıştır:

```bash
conda create -n imdb python=3.12
conda activate imdb
```

#### 4. Gerekli Kütüphaneleri Kur

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
pip install notebook ipykernel
python -m ipykernel install --user --name imdb --display-name "Python (imdb)"
```

> **Not:** GPU yoksa PyTorch'u şu şekilde kur (CPU versiyonu):
> ```bash
> pip install torch torchvision torchaudio
> ```

#### 5. Jupyter Notebook'u Başlat

```bash
cd imdb-sentiment
jupyter notebook
```

Tarayıcıda `http://localhost:8888` adresi açılacak.

#### 6. Notebookları Çalıştır

Açılan sayfada sırayla şu dosyaları aç ve hücreleri **Shift+Enter** ile çalıştır:

| Dosya | İçerik |
|-------|--------|
| `03_bilstm.ipynb` | BiLSTM modeli — veri yükleme, eğitim, değerlendirme |
| `04_distilbert.ipynb` | DistilBERT fine-tuning — eğitim, değerlendirme |

> **Önemli:** Her notebook'u açtıktan sonra sağ üstten kernel'ı **"Python (imdb)"** olarak seç.

#### 7. GPU Kontrolü

Notebook'ta şu kodu çalıştırarak GPU'nun çalışıp çalışmadığını kontrol edebilirsin:

```python
import torch
print(torch.cuda.is_available())       # True olmalı
print(torch.cuda.get_device_name(0))   # GPU adını gösterir
```

### Proje Yapısı

```
imdb-sentiment/
├── 03_bilstm.ipynb          # BiLSTM modeli
├── 04_distilbert.ipynb      # DistilBERT modeli
├── best_bilstm.pt           # Eğitilmiş BiLSTM ağırlıkları
├── bilstm_results.json      # BiLSTM sonuçları
├── distilbert_results.json  # DistilBERT sonuçları
├── requirements.txt         # Kütüphane listesi
├── figures/                 # Grafikler
│   ├── bilstm_results.png
│   ├── length_analysis.png
│   ├── distilbert_results.png
│   ├── distilbert_length.png
│   └── model_comparison.png
└── README.md
```

### Sık Karşılaşılan Sorunlar

**"No module named torch" hatası:**
Kernel'ı "Python (imdb)" olarak değiştir.

**"CUDA not available" uyarısı:**
GPU yoksa veya CUDA kurulu değilse model CPU'da çalışır, daha yavaş olur ama çalışır.

**"DatasetNotFoundError" hatası:**
`load_dataset("imdb")` kodunda tırnak içinde sadece `imdb` yazmalı, `imdb.csv` değil.

---

## 🇬🇧 English

### About

This project performs sentiment analysis on 50,000 IMDB movie reviews, comparing three machine learning approaches:

| Model | Accuracy | F1 Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| TF-IDF + Logistic Regression | 89.52% | 0.8951 | 0.9580 | ~2 min |
| TF-IDF + SVM | 90.10% | 0.9009 | 0.9601 | ~3 min |
| BiLSTM | 84.41% | 0.8433 | 0.9198 | 3.9 min |
| **DistilBERT** | **91.32%** | **0.9137** | **0.9724** | 20.3 min |

### Requirements

- Windows 10/11, macOS, or Linux
- Python 3.10, 3.11, or 3.12
- At least 8 GB RAM
- GPU (optional but recommended) — NVIDIA GPU + CUDA

### Installation — Step by Step

#### 1. Download and Install Miniconda

Miniconda is a free tool for managing Python environments.

👉 Download the **Windows 64-bit** installer from https://docs.anaconda.com/miniconda/

During installation:
- Select **"Just Me (recommended)"**
- Check **"Add to PATH"**

#### 2. Download the Project

**Option A — Using Git (recommended):**
```bash
git clone https://github.com/edipgenc/imdb-sentiment.git
cd imdb-sentiment
```

**Option B — As ZIP:**
- Click **Code → Download ZIP** at the top right of this page
- Extract the ZIP to a folder
- Open that folder in your terminal

#### 3. Create the Conda Environment

Open **Anaconda Prompt** from the Start menu and run:

```bash
conda create -n imdb python=3.12
conda activate imdb
```

#### 4. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn nltk
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers datasets accelerate
pip install notebook ipykernel
python -m ipykernel install --user --name imdb --display-name "Python (imdb)"
```

> **Note:** If you don't have a GPU, install the CPU version of PyTorch instead:
> ```bash
> pip install torch torchvision torchaudio
> ```

#### 5. Start Jupyter Notebook

```bash
cd imdb-sentiment
jupyter notebook
```

Your browser will open at `http://localhost:8888`.

#### 6. Run the Notebooks

Open the following files in order and run each cell with **Shift+Enter**:

| File | Content |
|------|---------|
| `03_bilstm.ipynb` | BiLSTM model — data loading, training, evaluation |
| `04_distilbert.ipynb` | DistilBERT fine-tuning — training, evaluation |

> **Important:** After opening each notebook, select **"Python (imdb)"** as the kernel from the top right.

#### 7. Verify GPU

Run this code in a notebook cell to check if your GPU is working:

```python
import torch
print(torch.cuda.is_available())       # Should print True
print(torch.cuda.get_device_name(0))   # Shows your GPU name
```

### Project Structure

```
imdb-sentiment/
├── 03_bilstm.ipynb          # BiLSTM model
├── 04_distilbert.ipynb      # DistilBERT model
├── best_bilstm.pt           # Trained BiLSTM weights
├── bilstm_results.json      # BiLSTM results
├── distilbert_results.json  # DistilBERT results
├── requirements.txt         # Library list
├── figures/                 # Output figures
│   ├── bilstm_results.png
│   ├── length_analysis.png
│   ├── distilbert_results.png
│   ├── distilbert_length.png
│   └── model_comparison.png
└── README.md
```

### Troubleshooting

**"No module named torch" error:**
Switch the kernel to "Python (imdb)".

**"CUDA not available" warning:**
If you don't have a GPU or CUDA isn't installed, the model will run on CPU — slower but functional.

**"DatasetNotFoundError" error:**
Make sure `load_dataset("imdb")` has only `imdb` in quotes, not `imdb.csv`.

---

### Hardware Used

| Component | Spec |
|-----------|------|
| CPU | Intel Core i7H |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 3050 Ti Laptop (4 GB VRAM) |
| CUDA | 13.2 |
| OS | Windows 11 |

---

### Author

**Edip GENÇ** — Ahmet Yesevi University, Master's Program in Artificial Intelligence

Supervisor: Prof. Dr. Abdurrahim TOKTAŞ
