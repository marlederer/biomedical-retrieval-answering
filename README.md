# biomedical-retrieval-answering

This project implements and evaluates different retrieval and re-ranking methods for biomedical question answering, specifically targeting the BioASQ challenge. It includes implementations for BM25, Dense Retrieval, and a KNRM re-ranker.

## Installation

1.  Clone the repository:
    ```bash
    git clone git@github.com:marlederer/biomedical-retrieval-answering.git
    cd biomedical-retrieval-answering
    ```
2.  Create a virtual environment (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate.bat`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

### BM25

To run the BM25 retrieval model, navigate to the `BM25` directory and execute the `main.py` script:

```bash
cd BM25
python main.py
```


### Dense Retrieval

To run the Dense Retrieval model, navigate to the `Dense` directory and execute the `dense_pipeline.py` script:

```bash
python -m spacy download en_core_web_sm
cd Dense
python bm25_dense_pipeline.py
```

### Re-ranker

The model for the re-ranker should be present in the `bioasq_reranker/models/` directory. If you need to train the re-ranker, use the `train.py` script in the same directory. You can change the training dataset in `bioasq_reranker/config.py`
```bash
cd bioasq_reranker
python train.py
```

To run the KNRM re-ranker, execute the `inference.py` script. You will need to provide the results from a retrieval model (like BM25 or Dense Retrieval) as input.

```bash
python bioasq_reranker/inference.py --input_file <path_to_retrieval_results.json> --model_path <path_to_model> --output_file <path_to_reranked_results.json>  --vocab_path <path_to_vocab.json> --ground_truth_file <path_to_ground_truth.json>
```
Replace the placeholders with the actual file paths for the following:

- The JSON file that contains the retrieval results
- The model file
- The desired name and path for the output file
- The vocabulary file path
- The ground truth file

One setting which should work out of the box is the following command.

```bash
python bioasq_reranker/inference.py --input_file bioasq_reranker/data/bm25.json --model_path bioasq_reranker/models/knrm_model.pth --output_file bioasq_reranker/reranked_bm25.json --vocab_path bioasq_reranker/data/vocab.json --ground_truth_file bioasq_reranker/data/training13b.json
```
