# 🌿 NutriLeafNet

**A Hybrid Deep Learning Framework for Intelligent Detection and Classification of Nutrient Deficiencies in Cassava Leaves**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/NutriLeafNet/blob/main/NutriLeafNet.ipynb)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ViT-ee4c2c)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b)
![License](https://img.shields.io/badge/license-MIT-green)

NutriLeafNet classifies nutrient deficiencies in cassava leaves from real field photographs using a **pure Vision Transformer (ViT)**. It uses **no convolutional backbone** (no ResNet, VGG, EfficientNet or MobileNet). The project is fully software-based and needs no hardware.

---

## ✨ Features

- **Pure transformer model**: `google/vit-base-patch16-224` splits the leaf image into 16×16 patches and learns global features with self-attention.
- **Real-world robustness**: brightness/contrast jitter, rotation, flips, random crops, Gaussian noise and blur simulate field lighting, camera angles and sensor quality.
- **Stratified 70 / 15 / 15** train / validation / test split, so every class keeps its proportion.
- **Sound training recipe**: AdamW, class-weighted cross-entropy with label smoothing, cosine schedule with warm-up, EarlyStopping and ModelCheckpoint on validation macro-F1.
- **Full evaluation**: accuracy, precision, recall, F1 and a confusion matrix on the held-out test set.
- **Explainability**: attention-rollout heatmaps (used instead of Grad-CAM, which needs convolutional feature maps).
- **Streamlit app**: upload a leaf photo and get the predicted class, a confidence score and an attention heatmap.

## 🧠 Method

```
Field photo ─► augment (train only) ─► 16×16 patch embedding ─► 12 × Transformer encoder (self-attention)
            ─► [CLS] token ─► linear head ─► deficiency class + confidence
                                  └─► attention rollout ─► heatmap
```

| Component | Choice |
|---|---|
| Backbone | `google/vit-base-patch16-224` (ImageNet-pretrained, fine-tuned) |
| Optimizer | AdamW (backbone LR 2e-5, head LR 1e-3, weight decay 0.05) |
| Loss | Cross-entropy, class weights, label smoothing 0.1 |
| Scheduler | Cosine decay with 10% warm-up |
| Callbacks | EarlyStopping (patience 5) and ModelCheckpoint on validation macro-F1 |
| Explainability | Attention rollout (Abnar & Zuidema, 2020) |

## 📁 Repository structure

```
NutriLeafNet/
├── NutriLeafNet.ipynb     # Colab notebook: data → train → evaluate → explain
├── nutrileafnet.py        # Same pipeline as a plain script (# %% cells)
├── app.py                 # Streamlit inference app
├── requirements.txt
└── README.md
```

## 🗂️ Dataset

The dataset is **not included** in this repository. Use real, field-captured cassava leaf photographs with expert-verified labels, organised one folder per class:

```
cassava_nutrient_dataset/
├── healthy/
├── nitrogen_deficiency/
├── potassium_deficiency/
└── magnesium_deficiency/      # your own classes
```

> **Note:** At the time of writing I could not verify a public multi-class cassava *nutrient-deficiency* image dataset. Related field data exists, such as the IITA cassava images in Ramcharan et al. (2019), which include a nutrient-deficiency class. Make sure your labels come from an agronomist or other expert, and describe your data source honestly in any report.

## 🚀 Quick start

### 1. Train in Google Colab (recommended)
1. Click the **Open in Colab** badge, or upload `NutriLeafNet.ipynb`.
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Edit `data_dir` and `out_dir` in the *Configuration* cell to point to your Drive folders.
4. Run all cells. Outputs are saved to `out_dir`:
   `best_model/`, `training_curves.png`, `confusion_matrix.png`, `attention_maps.png`, `test_metrics.json`.

### 2. Run the app locally
```bash
git clone https://github.com/YOUR_USERNAME/NutriLeafNet.git
cd NutriLeafNet
pip install -r requirements.txt

# copy the trained `best_model/` folder from Drive into this directory, then:
streamlit run app.py
```
To keep the model elsewhere: `NUTRILEAF_MODEL_DIR=/path/to/best_model streamlit run app.py`

## 📊 Results

*Fill this in after training on your dataset. Do not report numbers you have not measured.*

| Metric (test set) | Value |
|---|---|
| Accuracy | – |
| Macro precision | – |
| Macro recall | – |
| Macro F1 | – |

Add `confusion_matrix.png` and `attention_maps.png` here, for example `![Confusion matrix](docs/confusion_matrix.png)`.

## ⚠️ Limitations

- Attention maps show *where* the model looked, not proof of *why* it decided.
- Visual symptoms of different deficiencies (e.g. nitrogen vs. potassium) overlap, so expect some confusion between those classes.
- Accuracy depends on label quality and on how well the dataset covers varieties, growth stages, lighting and backgrounds.
- This is a research and educational prototype. It is not a substitute for soil or tissue testing or an agronomist's advice.

## 📚 References

- Dosovitskiy et al., *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021.
- Abnar & Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020.
- Ramcharan et al., *A Mobile-Based Deep Learning Model for Cassava Disease Diagnosis*, Frontiers in Plant Science, 2019. doi:10.3389/fpls.2019.00272
- Hugging Face `transformers`, `google/vit-base-patch16-224`.

## 👤 Author

**Kowshick**, B.E. Electronics and Communication Engineering, Sathyabama Institute of Science and Technology
Final-year project.

## 📄 License

Released under the MIT License. See `LICENSE`.
