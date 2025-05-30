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
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
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
Make sure the necessary data files are present in the `data/` directory as specified in the configuration.

### Dense Retrieval

To run the Dense Retrieval model, navigate to the `Dense` directory and execute the `dense_pipeline.py` script:

```bash
cd Dense
python dense_pipeline.py
```
Ensure that the pre-trained models and data are correctly set up as per the instructions in the `Dense/README.md` if available, or check the configuration files.

### Re-ranker

To run the KNRM re-ranker, navigate to the `bioasq_reranker` directory and execute the `inference.py` script. You will typically need to provide the results from a retrieval model (like BM25 or Dense Retrieval) as input.

```bash
cd bioasq_reranker
python inference.py --input_file <path_to_retrieval_results.json> --output_file <path_to_reranked_results.json>
```
Replace `<path_to_retrieval_results.json>` with the actual path to the JSON file containing the retrieval results and `<path_to_reranked_results.json>` with the desired output file name.
The model for the re-ranker should be present in the `bioasq_reranker/models/` directory. If you need to train the re-ranker, use the `train.py` script in the same directory.