'''
Training script for the KNRM reranker.
'''
import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json

import config
from knrm import KNRM
from data_loader import get_dataloaders, Vocabulary
from utils import tokenize_text


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def train_knrm():
    # Ensure model save directory exists
    ensure_dir(os.path.dirname(config.SAVE_MODEL_PATH))
    ensure_dir(os.path.dirname(config.VOCAB_PATH))
    ensure_dir(os.path.dirname(config.TRAIN_DATA_PATH))

    # --- 1. Load or Build Vocabulary ---
    print("Loading/Building vocabulary...")
    try:
        vocab = Vocabulary.load(config.VOCAB_PATH)
        print(f"Vocabulary loaded from {config.VOCAB_PATH}. Size: {len(vocab)}")
    except FileNotFoundError:
        print(f"Vocabulary file not found at {config.VOCAB_PATH}. Building from training data...")
        try:
            from data_loader import load_bioasq_data
            raw_train_data = load_bioasq_data(config.TRAIN_DATA_PATH)
            if not raw_train_data:
                print(f"ERROR: Training data not found at {config.TRAIN_DATA_PATH} or is empty. Cannot build vocabulary.")
                return
            
            all_training_texts = []
            for q_data in raw_train_data:
                all_training_texts.append(tokenize_text(q_data['body']))
                for snippet in q_data.get('snippets', []):
                    if snippet.get('text'):
                         all_training_texts.append(tokenize_text(snippet['text']))
            
            if not all_training_texts:
                print("ERROR: No text found in training data to build vocabulary.")
                return

            vocab = Vocabulary()
            vocab.build_vocab(all_training_texts, min_freq=config.MIN_WORD_FREQ)
            vocab.save(config.VOCAB_PATH)
            print(f"Vocabulary built and saved to {config.VOCAB_PATH}. Size: {len(vocab)}")
        except FileNotFoundError:
            print(f"ERROR: Training data file {config.TRAIN_DATA_PATH} not found. Please run data_loader.py first or ensure data exists.")
            return
        except Exception as e:
            print(f"An error occurred during fallback vocabulary creation: {e}")
            return

    # --- 2. Initialize Model ---
    print("Initializing KNRM model...")
    model = KNRM(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        n_kernels=config.N_KERNELS
    ).to(config.DEVICE)
    print(model)

    # --- 3. Prepare DataLoaders ---
    print("Preparing DataLoaders...")
    try:
        train_dataloader, val_dataloader = get_dataloaders(vocab, batch_size=config.BATCH_SIZE)
    except ValueError as e:
        print(f"Error creating dataloaders: {e}")
        print("This might be due to missing or empty training data. Ensure TRAIN_DATA_PATH in config.py is correct and the file is populated.")
        print(f"Attempted to load from: {config.TRAIN_DATA_PATH}")
        return
    
    if not train_dataloader:
        print("Failed to create train_dataloader. Exiting.")
        return

    # --- 4. Optimizer and Loss Function ---
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MarginRankingLoss(margin=config.MARGIN_LOSS)

    # --- 5. Training Loop ---
    print(f"Starting training for {config.NUM_EPOCHS} epochs on {config.DEVICE}...")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")

        for batch in train_progress_bar:
            optimizer.zero_grad()

            query_ids = batch['query_ids'].to(config.DEVICE)
            query_mask = batch['query_mask'].to(config.DEVICE)
            rel_doc_ids = batch['rel_doc_ids'].to(config.DEVICE)
            rel_doc_mask = batch['rel_doc_mask'].to(config.DEVICE)
            non_rel_doc_ids = batch['non_rel_doc_ids'].to(config.DEVICE)
            non_rel_doc_mask = batch['non_rel_doc_mask'].to(config.DEVICE)

            # Scores for relevant documents
            s_rel = model(query_ids, rel_doc_ids, query_mask, rel_doc_mask)
            # Scores for non-relevant documents
            s_non_rel = model(query_ids, non_rel_doc_ids, query_mask, non_rel_doc_mask)

            # Target tensor: y=1 indicates s_rel should be greater than s_non_rel
            target = torch.ones_like(s_rel).to(config.DEVICE)

            loss = criterion(s_rel, s_non_rel, target)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")

        # --- 6. Validation Loop ---
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]")
            with torch.no_grad():
                for batch in val_progress_bar:
                    query_ids = batch['query_ids'].to(config.DEVICE)
                    query_mask = batch['query_mask'].to(config.DEVICE)
                    rel_doc_ids = batch['rel_doc_ids'].to(config.DEVICE)
                    rel_doc_mask = batch['rel_doc_mask'].to(config.DEVICE)
                    non_rel_doc_ids = batch['non_rel_doc_ids'].to(config.DEVICE)
                    non_rel_doc_mask = batch['non_rel_doc_mask'].to(config.DEVICE)

                    s_rel = model(query_ids, rel_doc_ids, query_mask, rel_doc_mask)
                    s_non_rel = model(query_ids, non_rel_doc_ids, query_mask, non_rel_doc_mask)
                    target = torch.ones_like(s_rel).to(config.DEVICE)
                    
                    loss = criterion(s_rel, s_non_rel, target)
                    total_val_loss += loss.item()
                    val_progress_bar.set_postfix(loss=loss.item())
            
            avg_val_loss = total_val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1} - Average Validation Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
                print(f"Model improved and saved to {config.SAVE_MODEL_PATH}")
        else:
            torch.save(model.state_dict(), config.SAVE_MODEL_PATH)
            print(f"Model saved to {config.SAVE_MODEL_PATH} (no validation performed)")

    print("Training complete.")

if __name__ == '__main__':
    ensure_dir("data")
    train_knrm()

