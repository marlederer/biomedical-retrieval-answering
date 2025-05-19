# BioASQ Neural Re-ranker & Document Retriever

This project implements a system for the BioASQ task, focusing on:
1.  Retrieving candidate documents from an external source (e.g., BioASQ PubMed service) using keyword-based queries.
2.  Re-ranking the retrieved documents and their passages using a transformer-based cross-encoder architecture.

## Project Structure

bioasq_reranker/
├── data/                     # Directory for BioASQ JSON data and other data files
│   └── training13b.json      # Example BioASQ training file (replace with your version)
│   └── hard_negatives.json   # Example output from hard negative generation
├── saved_models/             # Directory where trained models are saved
│   └── my_biomedbert_reranker_hardcoded/ # Example trained model
├── api_client.py             # Client for interacting with external APIs (e.g., BioASQ PubMed service)
├── data_loader.py            # Handles data loading, preprocessing, and PyTorch Dataset for training
├── generate_hard_negatives.py # Script to generate hard negative samples for training
├── inference.py              # Script for fetching documents and re-ranking them using a trained model
├── keyword_extractor.py      # Extracts keywords from queries for document retrieval
├── model_arch.py             # Defines the neural network model architecture for re-ranking
├── train.py                  # Script for training the re-ranker
├── requirements.txt          # Python package dependencies
└── README.md                 # This file

## Setup

1.  **Create a Python Environment:**
    It's recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    python -m venv venv_bioasq
    # On Linux/macOS
    source venv_bioasq/bin/activate
    # On Windows
    # venv_bioasq\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK Resources:**
    The `keyword_extractor.py` script uses NLTK. It will attempt to download necessary resources ('punkt', 'stopwords') automatically on first run if they are not found. Alternatively, you can download them manually:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4.  **Download BioASQ Data:**
    *   Register at the [BioASQ Participants Area](https://participants-area.bioasq.org/datasets/).
    *   Download a training dataset (e.g., `training13b.json`, `training11b.json`, etc.).
    *   Place the downloaded JSON file into the `data/` directory.

## Training the Re-ranker

The `train.py` script handles the training process for the cross-encoder re-ranking model.
```bash
python train.py \
    --bioasq_file data/training13b.json \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --output_dir saved_models/my_biomedbert_reranker \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --num_neg_samples 1 \
    --max_train_questions 100 # Optional: use a subset for quick testing, remove for full training
```

## Performing Inference (Document Retrieval & Re-ranking)

The `inference.py` script uses a trained model to:
1.  Take a query.
2.  Extract keywords from the query.
3.  Fetch candidate documents from an external source (e.g., BioASQ PubMed service) using these keywords.
4.  Split documents into passages (if configured).
5.  Re-rank the documents/passages using the trained cross-encoder model.

**Configuration for `inference.py` is primarily done by modifying the `INFERENCE_CONFIG` dictionary directly within the `inference.py` script.** This includes setting the model path, query, API endpoints, and other parameters.

To run inference, you modify `INFERENCE_CONFIG` in `inference.py` and then execute:
```bash
python inference.py
```
Check the `INFERENCE_CONFIG` section in `inference.py` for parameters like:
*   `model_path`: Path to the saved re-ranker model.
*   `query`: The question to process.
*   `bioasq_api_endpoint`: URL for the BioASQ PubMed service.
*   `num_candidates_per_combination`: How many documents to fetch for each keyword combination.
*   And other parameters related to passage processing and re-ranking.

## Important Considerations

*   **Negative Sampling for Training:** The `data_loader.py` might use random negative sampling. For significantly better re-ranking performance, implement or use hard negative mining. The `generate_hard_negatives.py` script is an example of how to start this process.
*   **Corpus for Negatives:** Random negative sampling benefits from a diverse corpus.
*   **Computational Resources:** Fine-tuning transformer models generally requires a GPU for reasonable training times.
*   **API Keys/Email:** If fetching directly from PubMed (not via BioASQ proxy), ensure you provide a valid email as per NCBI's requirements.

## Example Workflow Scripts

1.  **Generate Hard Negatives (Optional but Recommended for Better Training):**
    This script uses a first-stage retrieval (implicitly, the BioASQ API can be seen as one) to find documents that are retrieved for a question but are not gold-standard relevant, to be used as harder negative examples during training.
    ```bash
    python generate_hard_negatives.py \
        --bioasq_file data/training13b.json \
        --output_file data/hard_negatives.json \
        --top_k_retrieval 50 \
        --num_hard_negatives_per_query 5
    ```
    *(Note: You would then modify `train.py` or `data_loader.py` to use these hard negatives.)*

2.  **Run Inference (Document Retrieval & Re-ranking):**
    As mentioned above, configure `INFERENCE_CONFIG` within `inference.py` first.
    ```bash
    # 1. Modify INFERENCE_CONFIG in inference.py (e.g., set your query, model_path)
    # 2. Then run:
    python inference.py
    ```
    The script will output the re-ranked documents to the console.
```<!-- filepath: c:\Users\a872460\TU\4 Semester\AIR\biomedical-retrieval-answering\bioasq_reranker\README.md -->
# BioASQ Neural Re-ranker & Document Retriever

This project implements a system for the BioASQ task, focusing on:
1.  Retrieving candidate documents from an external source (e.g., BioASQ PubMed service) using keyword-based queries.
2.  Re-ranking the retrieved documents and their passages using a transformer-based cross-encoder architecture.

## Project Structure

bioasq_reranker/
├── data/                     # Directory for BioASQ JSON data and other data files
│   └── training13b.json      # Example BioASQ training file (replace with your version)
│   └── hard_negatives.json   # Example output from hard negative generation
├── saved_models/             # Directory where trained models are saved
│   └── my_biomedbert_reranker_hardcoded/ # Example trained model
├── api_client.py             # Client for interacting with external APIs (e.g., BioASQ PubMed service)
├── data_loader.py            # Handles data loading, preprocessing, and PyTorch Dataset for training
├── generate_hard_negatives.py # Script to generate hard negative samples for training
├── inference.py              # Script for fetching documents and re-ranking them using a trained model
├── keyword_extractor.py      # Extracts keywords from queries for document retrieval
├── model_arch.py             # Defines the neural network model architecture for re-ranking
├── train.py                  # Script for training the re-ranker
├── requirements.txt          # Python package dependencies
└── README.md                 # This file

## Setup

1.  **Create a Python Environment:**
    It's recommended to use a virtual environment (e.g., venv or conda).
    ```bash
    python -m venv venv_bioasq
    # On Linux/macOS
    source venv_bioasq/bin/activate
    # On Windows
    # venv_bioasq\Scripts\activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK Resources:**
    The `keyword_extractor.py` script uses NLTK. It will attempt to download necessary resources ('punkt', 'stopwords') automatically on first run if they are not found. Alternatively, you can download them manually:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    ```

4.  **Download BioASQ Data:**
    *   Register at the [BioASQ Participants Area](https://participants-area.bioasq.org/datasets/).
    *   Download a training dataset (e.g., `training13b.json`, `training11b.json`, etc.).
    *   Place the downloaded JSON file into the `data/` directory.

## Training the Re-ranker

The `train.py` script handles the training process for the cross-encoder re-ranking model.
```bash
python train.py \
    --bioasq_file data/training13b.json \
    --model_name "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" \
    --output_dir saved_models/my_biomedbert_reranker \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --max_seq_length 512 \
    --num_neg_samples 1 \
    --max_train_questions 100 # Optional: use a subset for quick testing, remove for full training
```

## Performing Inference (Document Retrieval & Re-ranking)

The `inference.py` script uses a trained model to:
1.  Take a query.
2.  Extract keywords from the query.
3.  Fetch candidate documents from an external source (e.g., BioASQ PubMed service) using these keywords.
4.  Split documents into passages (if configured).
5.  Re-rank the documents/passages using the trained cross-encoder model.

**Configuration for `inference.py` is primarily done by modifying the `INFERENCE_CONFIG` dictionary directly within the `inference.py` script.** This includes setting the model path, query, API endpoints, and other parameters.

To run inference, you modify `INFERENCE_CONFIG` in `inference.py` and then execute:
```bash
python inference.py
```
Check the `INFERENCE_CONFIG` section in `inference.py` for parameters like:
*   `model_path`: Path to the saved re-ranker model.
*   `query`: The question to process.
*   `bioasq_api_endpoint`: URL for the BioASQ PubMed service.
*   `num_candidates_per_combination`: How many documents to fetch for each keyword combination.
*   And other parameters related to passage processing and re-ranking.

## Important Considerations

*   **Negative Sampling for Training:** The `data_loader.py` might use random negative sampling. For significantly better re-ranking performance, implement or use hard negative mining. The `generate_hard_negatives.py` script is an example of how to start this process.
*   **Corpus for Negatives:** Random negative sampling benefits from a diverse corpus.
*   **Computational Resources:** Fine-tuning transformer models generally requires a GPU for reasonable training times.
*   **API Keys/Email:** If fetching directly from PubMed (not via BioASQ proxy), ensure you provide a valid email as per NCBI's requirements.

## Example Workflow Scripts

1.  **Generate Hard Negatives (Optional but Recommended for Better Training):**
    This script uses a first-stage retrieval (implicitly, the BioASQ API can be seen as one) to find documents that are retrieved for a question but are not gold-standard relevant, to be used as harder negative examples during training.
    ```bash
    python generate_hard_negatives.py \
        --bioasq_file data/training13b.json \
        --output_file data/hard_negatives.json \
        --top_k_retrieval 50 \
        --num_hard_negatives_per_query 5
    ```
    *(Note: You would then modify `train.py` or `data_loader.py` to use these hard negatives.)*

2.  **Run Inference (Document Retrieval & Re-ranking):**
    As mentioned above, configure `INFERENCE_CONFIG` within `inference.py` first.
    ```bash
    # 1. Modify INFERENCE_CONFIG in inference.py (e.g., set your query, model_path)
    # 2. Then run:
    python inference.py
    ```
    The script will output the re-ranked documents to the console.
