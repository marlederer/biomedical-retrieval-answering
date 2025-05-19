# train.py (with hardcoded parameters)
import os
import random
import logging
import json
from tqdm import tqdm # Moved import to top

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW # Changed import for AdamW

from transformers import get_linear_schedule_with_warmup # AdamW removed from here

from data_loader import (
    load_bioasq_json,
    build_passage_corpus_from_bioasq,
    create_training_samples,
    ReRankerDataset,
    load_hard_negatives
)
from model_arch import CrossEncoderReRanker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Hardcoded Training Configuration ---
TRAINING_CONFIG = {
    "bioasq_file": "data/training13b.json", # Path to BioASQ training JSON file
    "hard_negatives_file": "data/hard_negatives.json", # Optional path to hard negatives
    # "hard_negatives_file": None, # Set to None if not using hard negatives

    "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", # Hugging Face model
    "output_dir": "bioasq_reranker/saved_models/my_biomedbert_reranker_hardcoded", # Dir to save model

    "num_epochs": 5,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "max_seq_length": 512,
    "num_neg_samples": 2, # Total negative samples per positive (includes hard & random)
    "val_split_ratio": 0.1,
    "max_train_questions": None, # Max questions to use (e.g., 100 for quick test, None for all)
    "save_steps": 0, # Save checkpoint every X steps (0 for epoch/best-based saving)
    "seed": 42,

    # Dummy data creation for standalone testing (if main files are not present)
    "create_dummy_data_if_missing": True,
    "dummy_bioasq_file": "dummy_bioasq_for_train_hc.json",
    "dummy_hard_negs_file": "dummy_hard_negs_for_train_hc.json",
    "dummy_max_questions": 20
}
# --- End of Hardcoded Configuration ---

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def train_model(config): # Renamed from train to avoid conflict, now takes config dict
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    bioasq_file_path = config["bioasq_file"]
    hard_negs_file_path = config["hard_negatives_file"]

    # --- Handle Dummy Data Creation (for testing if real files are missing) ---
    if config["create_dummy_data_if_missing"]:
        if not os.path.exists(bioasq_file_path):
            logger.warning(f"BioASQ file {bioasq_file_path} not found. Creating a dummy file for demonstration.")
            bioasq_file_path = config["dummy_bioasq_file"]
            dummy_bioasq_content = {
                "questions": [{"id": f"q{i}", "body": f"What is item {i}?", "snippets": [{"text": f"Item {i} is a test entity."}]} for i in range(1, config["dummy_max_questions"] + 30)] # More varied IDs
            }
            with open(bioasq_file_path, 'w') as f: json.dump(dummy_bioasq_content, f)
            # If BioASQ is dummy, also make hard_negs dummy if it's specified but not found
            if hard_negs_file_path and not os.path.exists(hard_negs_file_path):
                 hard_negs_file_path = config["dummy_hard_negs_file"]
                 logger.info(f"Creating dummy hard negatives file at {hard_negs_file_path}")
                 dummy_hn_content = {f"q{i}": [f"This is a fake hard negative for item {i}."] for i in range(1, config["dummy_max_questions"] + 1)}
                 with open(hard_negs_file_path, 'w') as f: json.dump(dummy_hn_content, f)
            elif hard_negs_file_path is None: # If original config was None, keep it None
                pass # No hard negs file to make dummy for

            # Override max_train_questions if using dummy data and it's not already limited
            if config["max_train_questions"] is None or config["max_train_questions"] > config["dummy_max_questions"] :
                config["max_train_questions"] = config["dummy_max_questions"]
                logger.info(f"Using dummy data. max_train_questions set to {config['max_train_questions']}")
    # --- End of Dummy Data Handling ---


    # --- 1. Load and Prepare Data ---
    logger.info(f"Loading BioASQ data from: {bioasq_file_path}")
    raw_questions_data = load_bioasq_json(bioasq_file_path)
    if not raw_questions_data:
        logger.error("No questions data loaded. Exiting training.")
        return

    hard_negatives_map = None
    if hard_negs_file_path: # Use the potentially updated path
        logger.info(f"Loading hard negatives from: {hard_negs_file_path}")
        hard_negatives_map = load_hard_negatives(hard_negs_file_path)
    else:
        logger.info("No hard negatives file provided.")

    passage_corpus = build_passage_corpus_from_bioasq(raw_questions_data)
    if not passage_corpus:
        logger.warning("Passage corpus could not be built or is empty.")

    all_training_samples = create_training_samples(
        raw_questions_data,
        passage_corpus,
        config["num_neg_samples"],
        max_questions=config["max_train_questions"],
        hard_negatives_map=hard_negatives_map
    )
    if not all_training_samples:
        logger.error("No training samples were generated. Exiting.")
        return

    num_samples = len(all_training_samples)
    val_size = int(config["val_split_ratio"] * num_samples)
    train_size = num_samples - val_size

    if train_size <=0 or val_size <=0:
        logger.warning(f"Not enough samples for train/val split ({num_samples} total). Using all for training and skipping validation.")
        train_samples_list = all_training_samples
        val_samples_list = []
    else:
        random.shuffle(all_training_samples)
        train_samples_list = all_training_samples[:train_size]
        val_samples_list = all_training_samples[train_size:]

    logger.info(f"Number of training samples: {len(train_samples_list)}")
    logger.info(f"Number of validation samples: {len(val_samples_list)}")

    train_dataset = ReRankerDataset(train_samples_list, config["model_name"], config["max_seq_length"])
    val_dataset = None
    if val_samples_list:
        val_dataset = ReRankerDataset(val_samples_list, config["model_name"], config["max_seq_length"])

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty. Cannot proceed.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = None
    if val_dataset and len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # --- 2. Initialize Model, Optimizer, Scheduler ---
    logger.info(f"Initializing model: {config['model_name']}")
    model = CrossEncoderReRanker(config["model_name"])
    model.to(device)

    # Use torch.optim.AdamW
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], eps=1e-8)
    criterion = torch.nn.BCEWithLogitsLoss()

    total_steps = len(train_dataloader) * config["num_epochs"]
    if total_steps == 0 and config["num_epochs"] > 0:
        logger.warning("Train dataloader is empty, but epochs > 0. Total steps will be 0.")

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.05 * total_steps) if total_steps > 0 else 0,
                                                num_training_steps=total_steps if total_steps > 0 else 1)

    # --- 3. Training Loop ---
    logger.info("***** Starting Training *****")
    logger.info(f"  Num epochs = {config['num_epochs']}")
    logger.info(f"  Batch size = {config['batch_size']}")
    logger.info(f"  Total optimization steps = {total_steps}")

    if total_steps == 0 and config["num_epochs"] > 0:
        logger.error("Total optimization steps is 0. Training cannot proceed.")
        # Clean up dummy files if they were created by this run
        if config["create_dummy_data_if_missing"]:
            if os.path.exists(config["dummy_bioasq_file"]): os.remove(config["dummy_bioasq_file"])
            if config["dummy_hard_negs_file"] and os.path.exists(config["dummy_hard_negs_file"]): os.remove(config["dummy_hard_negs_file"])
        return

    global_step = 0
    best_val_loss = float('inf')
    output_dir = config["output_dir"] # Use the config value

    for epoch in range(config["num_epochs"]):
        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Train)", disable=len(train_dataloader) == 0)
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if total_steps > 0: scheduler.step()

            total_train_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({'loss': loss.item()})

            if config["save_steps"] > 0 and global_step % config["save_steps"] == 0:
                output_checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(output_checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_checkpoint_dir, "pytorch_model.bin"))
                logger.info(f"Saved model checkpoint to {output_checkpoint_dir}")

        if len(train_dataloader) > 0:
            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} - No training data processed.")

        # --- Validation Step ---
        if val_dataloader and len(val_dataloader) > 0:
            model.eval()
            total_val_loss = 0
            all_preds_probs = []
            all_true_labels = []

            logger.info(f"Running validation for Epoch {epoch+1}...")
            progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} (Val)")
            with torch.no_grad():
                for batch_val in progress_bar_val: # Renamed batch to batch_val
                    input_ids = batch_val['input_ids'].to(device)
                    attention_mask = batch_val['attention_mask'].to(device)
                    labels = batch_val['labels'].to(device).unsqueeze(1)

                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    all_preds_probs.extend(probs.cpu().numpy().flatten())
                    all_true_labels.extend(labels.cpu().numpy().flatten())
                    progress_bar_val.set_postfix({'val_loss': loss.item()})

            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            if all_preds_probs and all_true_labels:
                preds_binary = (np.array(all_preds_probs) > 0.5).astype(int)
                accuracy = np.mean(preds_binary == np.array(all_true_labels))
                logger.info(f"Epoch {epoch+1} Validation Accuracy (0.5 thresh): {accuracy:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {output_dir}")
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                model_to_save.bert.config.to_json_file(os.path.join(output_dir, "config.json"))
                train_dataset.tokenizer.save_pretrained(output_dir)
                # Save training config used for this run
                with open(os.path.join(output_dir, "training_config_hardcoded.json"), "w") as f:
                    # Create a serializable copy of config for JSON dump
                    serializable_config = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    json.dump(serializable_config, f, indent=4)

        elif config["save_steps"] <= 0: # No validation, save at end of epoch if not saving by steps
            logger.info(f"No validation or validation dataloader empty. Saving model at end of epoch {epoch+1} to {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            model_to_save.bert.config.to_json_file(os.path.join(output_dir, "config.json"))
            train_dataset.tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "training_config_hardcoded.json"), "w") as f:
                serializable_config = {k: v for k, v in config.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                json.dump(serializable_config, f, indent=4)


    logger.info("Training complete.")
    # Final save if not triggered by validation best or save_steps
    if not (val_dataloader and len(val_dataloader) > 0) and config["save_steps"] <= 0:
        logger.info(f"Final model saved to {output_dir} (end of training, no validation-based best or step saves).")

    # Clean up dummy files if they were created by this run
    if config["create_dummy_data_if_missing"]:
        dummy_bioasq_created_path = config["dummy_bioasq_file"]
        dummy_hard_negs_created_path = config["dummy_hard_negs_file"]

        # Check if the dummy files were actually used (paths might have been overridden)
        current_bioasq_is_dummy = (bioasq_file_path == dummy_bioasq_created_path)
        current_hard_negs_is_dummy = (hard_negs_file_path == dummy_hard_negs_created_path and hard_negs_file_path is not None)

        if current_bioasq_is_dummy and os.path.exists(dummy_bioasq_created_path):
            os.remove(dummy_bioasq_created_path)
            logger.info(f"Removed dummy bioasq data file: {dummy_bioasq_created_path}")
        if current_hard_negs_is_dummy and os.path.exists(dummy_hard_negs_created_path):
            os.remove(dummy_hard_negs_created_path)
            logger.info(f"Removed dummy hard negatives file: {dummy_hard_negs_created_path}")


if __name__ == "__main__":
    # Ensure the 'data' and 'saved_models' directories exist for dummy file creation and model saving
    # This is good practice even if not using dummy data.
    os.makedirs("data", exist_ok=True)
    os.makedirs(TRAINING_CONFIG["output_dir"], exist_ok=True) # Make sure output_dir exists

    train_model(TRAINING_CONFIG)
