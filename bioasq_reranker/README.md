# BioASQ Neural Re-ranker

This project implements a neural re-ranking model for the BioASQ task (specifically for passage/snippet retrieval), fulfilling the third challenge: "a neural re-ranking model". It uses a transformer-based cross-encoder architecture.

## Project Structure

bioasq_reranker/
├── data/ # Directory for BioASQ JSON data
│ └── training11b.json # Example BioASQ training file (replace with your version)
├── saved_models/ # Directory where trained models are saved
├── data_loader.py # Handles data loading, preprocessing, and PyTorch Dataset
├── model_arch.py # Defines the neural network model architecture
├── train.py # Script for training the re-ranker
├── inference.py # Script for using the trained model to re-rank passages
├── requirements.txt # Python package dependencies
└── README.md # This file


## Setup

1.  **Create a Python Environment:**
    It's recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    python -m venv venv_bioasq
    source venv_bioasq/bin/activate  # On Linux/macOS
    # venv_bioasq\Scripts\activate  # On Windows
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download BioASQ Data:**
    *   Register at the [BioASQ Participants Area](https://participants-area.bioasq.org/datasets/).
    *   Download a training dataset (e.g., `training11b.json`, `training10b.json`, etc.). The filenames might change slightly per version (e.g., `11bMeSH.json`). Adjust paths in scripts if needed.
    *   Place the downloaded JSON file (e.g., `training11b.json`) into the `data/` directory.

## Training the Re-ranker

The `train.py` script handles the training process.

```bash
python train.py \
    --bioasq_file data/training11b.json \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --output_dir saved_models/my_biomedbert_reranker \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --num_neg_samples 1 \
    --max_train_questions 100 # Optional: use a subset for quick testing, remove for full training

## Performing Inference
The inference.py script uses a trained model to re-rank a list of candidate passages for a given query.

python inference.py \
    --model_path saved_models/my_biomedbert_reranker \
    --query "What are the treatments for adenocarcinoma of the lung?" \
    --passages \
        "Adenocarcinoma treatment often involves surgery." \
        "Lung cancer can be of different types." \
        "Chemotherapy is a common option for advanced lung adenocarcinoma." \
        "Photosynthesis is a plant process."

Arguments for inference.py:
--model_path: Path to the directory containing the saved model and tokenizer (created by train.py).
--query: The question to re-rank passages for.
--passages: A list of candidate passages (strings).
--max_seq_length: (Optional) Max sequence length, should match training if possible.
--batch_size: (Optional) Inference batch size for re-ranking multiple passages.
Important Considerations
Negative Sampling: The current data_loader.py uses random negative sampling. For significantly better performance, implement hard negative mining. This involves using your first-stage retriever (e.g., BM25 or a bi-encoder) to find passages that are retrieved for a question but are not gold-standard relevant.
Corpus for Negatives: The current random negative sampling uses all snippets from the training data as a pseudo-corpus. A more robust approach would be to use a larger, more diverse corpus (e.g., all PubMed abstracts related to BioASQ).
Computational Resources: Fine-tuning transformer models requires a GPU for reasonable training times.
Full BioASQ Document Processing: This example primarily focuses on re-ranking provided snippets. For document-level re-ranking, you would need to process the full documents linked in the BioASQ data (often PubMed articles), potentially segmenting them into passages.


## Running the files

python generate_hard_negatives.py \
    --bioasq_file data/training11b.json \
    --output_file data/hard_negatives.json \
    --top_k_retrieval 50 \
    --num_hard_negatives_per_query 5

