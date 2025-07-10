# 🐔 HealthyHen - Klasifikasi Penyakit Ayam dari Feses

HealthyHen adalah sistem klasifikasi citra berbasis deep learning yang mampu mengenali 4 kondisi umum feses ayam untuk mendeteksi penyakit secara dini.

## 📂 Struktur Direktori

```
├── model/
│   └── healthyHen_model.tflite         # Model akhir dalam format TFLite (siap deploy)
├── sample_data/
│   ├── train/
│   ├── val/
│   └── test/                           # Folder data (per kelas): Coccidiosis, Healthy, etc
├── healthyHen_notebook.ipynb          # Notebook utama training & evaluasi
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## 🧬 Arsitektur Model

Model menggunakan transfer learning dengan arsitektur:

* Base: `EfficientNetV2S` (pretrained ImageNet)
* Head:

  * `GlobalAveragePooling2D`
  * `Dense(128) → BatchNorm → ReLU → Dropout(0.3)`
  * `Dense(4, activation='softmax')`

Model dikompilasi menggunakan `Adam`, `categorical_crossentropy`, dan metrik `accuracy`. Fine-tuning dilakukan dengan learning rate rendah.

## 🗂️ Dataset

Dataset yang digunakan merupakan salinan langsung dari:

> Jayavrinda Vrindavanam, Pradeep Kumar, Gaurav Kamath, Chandrashekar N, and Govind Patil. (2023). *Poultry Pathology Visual Dataset*. Kaggle. [https://doi.org/10.34740/KAGGLE/DS/3951043](https://doi.org/10.34740/KAGGLE/DS/3951043)

Total kelas:

* Coccidiosis
* Healthy
* Newcastle Disease
* Salmonella

Resolusi citra telah distandarisasi ke `224x224` piksel.

## 🚀 Cara Menjalankan (Training)

1. Buat virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Buka dan jalankan `healthyHen_notebook.ipynb`

## 🧪 Evaluasi

Model mencapai akurasi validasi >90% pada data subset yang disampling. Hasil ditampilkan dalam notebook melalui confusion matrix dan classification report.

## 📱 Deploy TFLite

Model akhir tersedia di folder `model/healthyHen_model.tflite` dan dapat digunakan untuk deployment di perangkat edge/mobile.

Contoh penggunaan akan tersedia pada rilis berikutnya.

---

Beyond the Outliers
Datahon 2025
