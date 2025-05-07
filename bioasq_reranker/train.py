import argparse
import os
import random
import logging
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer

from data_loader import load_bioasq_json, build_passage_corpus_from_bioasq, create_training_samples, ReRankerDataset
from model_arch import CrossEncoderReRanker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load and Prepare Data ---
    logger.info(f"Loading BioASQ data from: {args.bioasq_file}")
    raw_questions_data = load_bioasq_json(args.bioasq_file)
    if not raw_questions_data:
        logger.error("No questions data loaded. Exiting training.")
        return

    passage_corpus = build_passage_corpus_from_bioasq(raw_questions_data)
    if not passage_corpus:
        logger.error("Passage corpus could not be built. This is essential for negative sampling. Exiting.")
        return
        
    all_training_samples = create_training_samples(
        raw_questions_data,
        passage_corpus,
        args.num_neg_samples,
        max_questions=args.max_train_questions
    )
    if not all_training_samples:
        logger.error("No training samples were generated. Check your data and parameters. Exiting.")
        return

    # Split data into training and validation
    num_samples = len(all_training_samples)
    val_size = int(args.val_split_ratio * num_samples)
    train_size = num_samples - val_size
    
    if train_size <=0 or val_size <=0:
        logger.warning(f"Not enough samples for train/val split ({num_samples} total). Using all for training and skipping validation.")
        train_samples = all_training_samples
        val_samples = [] # No validation
    else:
        train_samples, val_samples = random_split(all_training_samples, [train_size, val_size])
    
    logger.info(f"Number of training samples: {len(train_samples)}")
    logger.info(f"Number of validation samples: {len(val_samples)}")

    train_dataset = ReRankerDataset(list(train_samples), args.model_name, args.max_seq_length)
    if val_samples:
        val_dataset = ReRankerDataset(list(val_samples), args.model_name, args.max_seq_length)
    else:
        val_dataset = None

    if len(train_dataset) == 0:
        logger.error("Train dataset is empty. Cannot proceed with training.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if val_dataset and len(val_dataset) > 0:
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        val_dataloader = None


    # --- 2. Initialize Model, Optimizer, Scheduler ---
    logger.info(f"Initializing model: {args.model_name}")
    model = CrossEncoderReRanker(args.model_name)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    criterion = torch.nn.BCEWithLogitsLoss() # Suitable for single logit output

    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(0.05 * total_steps), # 5% warmup
                                                num_training_steps=total_steps)

    # --- 3. Training Loop ---
    logger.info("***** Starting Training *****")
    logger.info(f"  Num epochs = {args.num_epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Train)")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1) # Match shape for BCEWithLogitsLoss

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1
            
            progress_bar.set_postfix({'loss': loss.item()})

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(output_checkpoint_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(output_checkpoint_dir, "pytorch_model.bin"))
                # model_to_save.bert.config.to_json_file(os.path.join(output_checkpoint_dir, "config.json")) # Save full config for BERT part
                # train_dataset.tokenizer.save_pretrained(output_checkpoint_dir) # Save tokenizer
                logger.info(f"Saved model checkpoint to {output_checkpoint_dir}")


        avg_train_loss = total_train_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

        # --- Validation Step ---
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            all_preds_probs = []
            all_true_labels = []
            
            logger.info(f"Running validation for Epoch {epoch+1}...")
            progress_bar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs} (Val)")
            with torch.no_grad():
                for batch in progress_bar_val:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device).unsqueeze(1)

                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    total_val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    all_preds_probs.extend(probs.cpu().numpy().flatten())
                    all_true_labels.extend(labels.cpu().numpy().flatten())
                    progress_bar_val.set_postfix({'val_loss': loss.item()})


            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")

            # Simple accuracy for binary classification (threshold at 0.5)
            if all_preds_probs and all_true_labels:
                preds_binary = (np.array(all_preds_probs) > 0.5).astype(int)
                accuracy = np.mean(preds_binary == np.array(all_true_labels))
                logger.info(f"Epoch {epoch+1} Validation Accuracy (0.5 thresh): {accuracy:.4f}")
                # from sklearn.metrics import roc_auc_score # Can add AUC if needed
                # if len(np.unique(all_true_labels)) > 1: # AUC requires at least two classes
                #    auc = roc_auc_score(all_true_labels, all_preds_probs)
                #    logger.info(f"Epoch {epoch+1} Validation AUC: {auc:.4f}")


            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Saving model to {args.output_dir}")
                os.makedirs(args.output_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model # For DataParallel
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                # Save the BERT part config and tokenizer for easier reloading with AutoModel
                model_to_save.bert.config.to_json_file(os.path.join(args.output_dir, "config.json"))
                train_dataset.tokenizer.save_pretrained(args.output_dir)
                
                # Save training arguments
                with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4)
        else: # No validation, save at end of each epoch or based on save_steps only
            if args.save_steps <= 0: # If not saving by steps, save at end of epoch
                logger.info(f"No validation. Saving model at end of epoch {epoch+1} to {args.output_dir}")
                os.makedirs(args.output_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
                model_to_save.bert.config.to_json_file(os.path.join(args.output_dir, "config.json"))
                train_dataset.tokenizer.save_pretrained(args.output_dir)
                with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                    json.dump(vars(args), f, indent=4)


    logger.info("Training complete.")
    if not val_dataloader and args.save_steps <= 0: # Ensure model is saved if not saved by other conditions
        logger.info(f"Final model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BioASQ Neural Re-ranker")
    parser.add_argument("--bioasq_file", type=str, required=True, help="Path to BioASQ training JSON file")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", help="Hugging Face model identifier")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trained model and tokenizer")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Optimizer learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--num_neg_samples", type=int, default=1, help="Number of random negative samples per positive sample")
    parser.add_argument("--val_split_ratio", type=float, default=0.1, help="Ratio of data for validation")
    parser.add_argument("--max_train_questions", type=int, default=None, help="Max questions from BioASQ to use (for quick test/debug)")
    parser.add_argument("--save_steps", type=int, default=0, help="Save checkpoint every X updates steps. 0 for no step-based saving (only epoch/best).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Create a dummy BioASQ file if the specified one doesn't exist for quick testing
    # (This is just for making the script runnable without immediate data download for users)
    if not os.path.exists(args.bioasq_file) and "dummy_bioasq_for_train.json" not in args.bioasq_file:
        logger.warning(f"BioASQ file {args.bioasq_file} not found. Creating a dummy file for demonstration.")
        args.bioasq_file = "dummy_bioasq_for_train.json"
        dummy_bioasq_content = {
            "questions": [
                {"id": f"q{i}", "body": f"What is item {i}?", "snippets": [{"text": f"Item {i} is a test entity."},{"text": f"More about item {i} here."}]} for i in range(1, 21)
            ] + [ # Add some questions with no snippets to test robustness
                {"id": f"qn{j}", "body": f"Question {j} with no snippets?", "documents": ["doc_url"]} for j in range(1, 6)
            ]
        }
        with open(args.bioasq_file, 'w') as f:
            json.dump(dummy_bioasq_content, f)
        args.max_train_questions = 20 # Use a small number for dummy data
        logger.info(f"Using dummy data from {args.bioasq_file} with max_train_questions={args.max_train_questions}")


    train(args)

    # Clean up dummy file if created
    if "dummy_bioasq_for_train.json" in args.bioasq_file and os.path.exists(args.bioasq_file):
        os.remove(args.bioasq_file)
        logger.info(f"Removed dummy data file: {args.bioasq_file}")