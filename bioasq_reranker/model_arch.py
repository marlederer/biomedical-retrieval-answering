import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReRanker(nn.Module):
    def __init__(self, model_name_or_path):
        super(CrossEncoderReRanker, self).__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name_or_path)
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.classifier = nn.Linear(config.hidden_size, 1) # Output 1 logit for binary classification
            logger.info(f"CrossEncoderReRanker initialized with model: {model_name_or_path}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name_or_path}: {e}")
            raise

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token's representation for classification
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

# --- Example Usage (for testing this file standalone) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with a common model
    model_name = "bert-base-uncased" # Use a smaller model for quick testing
    # model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" # For actual use

    try:
        model = CrossEncoderReRanker(model_name)
        logger.info("Model instantiated successfully.")

        # Dummy input
        batch_size = 2
        seq_len = 64
        dummy_input_ids = torch.randint(0, model.bert.config.vocab_size, (batch_size, seq_len))
        dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

        with torch.no_grad():
            logits = model(dummy_input_ids, dummy_attention_mask)
        
        logger.info(f"Output logits shape: {logits.shape}") # Expected: (batch_size, 1)
        assert logits.shape == (batch_size, 1)
        logger.info("Model forward pass test successful.")

    except Exception as e:
        logger.error(f"Error during model_arch.py test: {e}")