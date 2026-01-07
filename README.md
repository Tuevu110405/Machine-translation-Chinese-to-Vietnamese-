# Machine-translation-Chinese-to-Vietnamese-

This project successfully implements a basic model that translates Chinese into Vietnamese. This model use Transformers architecture, and it is trained with contrastive learning to improve semantic alignment 

## Key Feature

* **Transformer architecture:**
    * **Rotary Positional Encoding:** better sequence processing than traditional positional encoding 
    * **RMSNorm:** more stable and less computation than Layer Norm
* **SwiGLU Activation:** enhancing the feature representation.
    * **Grouped Query Attention (GQA):** Optimizing cache and inference time.
* **2-Stage Training:**
    * **Stage 1 (Warm-up):** Training with Label Smoothing loss.
    * **Stage 2 (Fine-tuning):** Featuring **Contrastive Learning (InfoNCE Loss)** + CE Loss. This training leads model to improve the ability of understanding sematic similarity by draging semantic representations of parallel sentence closer.
* **Data & Tokenizer:**
    * **SentencePiece (BPE):** Shared vocabulary (8000 tokens) for both languages.
    * **Sliding Window Sampling:** The technique of reverse sampling (Vi->Zh) using sliding windows is employed to balance the data per epoch.

## Weights
You can get all save weights here [Weights](https://drive.google.com/drive/folders/1wK92Cf532dBq_FQ51xuc5CP3lhpwVPlZ?usp=sharing)
    
## Results 
| Model Variant | SacreBLEU (Zh→Vi) | Test Loss | Download Link |
| :--- | :---: | :---: | :---: |
| **Stage 1 (Base)** | 18.14 | 2.9329 (Label Smoothing) | [Link](https://drive.google.com/file/d/1goJr_jguN_h_6dxp--gwq-ye2aUEj2Bn/view?usp=drive_link) |
| **Stage 2 (Contrastive)** | 24.28 | 3.1104 (InfoNCE + Label Smoothing) | [Link](https://drive.google.com/file/d/1rvBcCdTcBI6G_CTIGRudnoBW19K_T9uJ/view?usp=drive_link) |
