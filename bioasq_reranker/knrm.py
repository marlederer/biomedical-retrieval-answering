'''
Implementation of the Kernel-based Neural Ranking Model (KNRM).
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class KNRM(nn.Module):
    '''
    Kernel-based Neural Ranking Model (KNRM).

    Args:
        embedding_matrix (torch.Tensor): Pre-trained word embedding matrix.
        n_kernels (int): Number of RBF kernels to use.
        mu_init (list or torch.Tensor, optional): Initial means for RBF kernels. 
                                                 Defaults to linearly spaced values in [-0.9, 0.9].
        sigma_init (list or torch.Tensor, optional): Initial sigmas for RBF kernels. 
                                                   Defaults to a fixed value (e.g., 0.1) for all kernels.
        train_embeddings (bool, optional): Whether to fine-tune embeddings. Defaults to True.
        embedding_dim (int, optional): Dimension of embeddings, required if embedding_matrix is None.
        vocab_size (int, optional): Size of vocabulary, required if embedding_matrix is None and creating new embeddings.
    '''
    def __init__(self, embedding_matrix=None, vocab_size=None, embedding_dim=300, 
                 n_kernels=11, mu_init=None, sigma_init=None, train_embeddings=True):
        super(KNRM, self).__init__()

        if embedding_matrix is not None:
            self.vocab_size, self.embedding_dim = embedding_matrix.shape
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=not train_embeddings)
        elif vocab_size is not None and embedding_dim is not None:
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
            # nn.init.xavier_uniform_(self.embedding.weight) 
        else:
            raise ValueError("Either embedding_matrix or (vocab_size and embedding_dim) must be provided.")

        self.n_kernels = n_kernels

        # Initialize RBF kernel parameters (mu and sigma)
        if mu_init is None:
            # Linearly spaced means from -0.9 to 0.9 (covering cosine similarities from near -1 to near 1 after some initial non-linearity)
            # The first kernel is exact match (mu=1.0), others are spread out.
            mus = [-0.9 + (1.8 / (n_kernels - 2)) * i for i in range(n_kernels -1)]
            mus.append(0.999) 
            self.mu = nn.Parameter(torch.tensor(mus, dtype=torch.float32).view(1, 1, 1, n_kernels))
        else:
            if not isinstance(mu_init, torch.Tensor):
                mu_init = torch.tensor(mu_init, dtype=torch.float32)
            self.mu = nn.Parameter(mu_init.view(1, 1, 1, n_kernels))

        if sigma_init is None:
            # A common small sigma for all kernels, or make them different
            sigmas = [0.1] * (n_kernels - 1) + [0.001] # Smaller sigma for the exact match kernel
            self.sigma = nn.Parameter(torch.tensor(sigmas, dtype=torch.float32).view(1, 1, 1, n_kernels))
        else:
            if not isinstance(sigma_init, torch.Tensor):
                sigma_init = torch.tensor(sigma_init, dtype=torch.float32)
            self.sigma = nn.Parameter(sigma_init.view(1, 1, 1, n_kernels))

        # Final dense layer to combine kernel features
        # Input dimension is n_kernels, output is 1 (the relevance score)
        self.dense_layer = nn.Linear(self.n_kernels, 1)

    def _compute_match_matrix(self, query_embed, doc_embed, query_mask=None, doc_mask=None):
        '''
        Computes the cosine similarity match matrix between query and document embeddings.
        query_embed: [batch_size, query_seq_len, emb_dim]
        doc_embed: [batch_size, doc_seq_len, emb_dim]
        query_mask: [batch_size, query_seq_len] (optional, 1 for tokens, 0 for padding)
        doc_mask: [batch_size, doc_seq_len] (optional, 1 for tokens, 0 for padding)
        Returns: [batch_size, query_seq_len, doc_seq_len]
        '''
        # Normalize embeddings for cosine similarity
        query_norm = F.normalize(query_embed, p=2, dim=2)
        doc_norm = F.normalize(doc_embed, p=2, dim=2)

        # Cosine similarity matrix using batch matrix multiplication
        # (batch, q_len, emb_dim) x (batch, emb_dim, d_len) -> (batch, q_len, d_len)
        match_matrix = torch.bmm(query_norm, doc_norm.transpose(1, 2))
        
        # Apply mask to handle padding if necessary (set similarity for padded terms to a very low value)
        if query_mask is not None and doc_mask is not None:
            # Expand masks to be compatible with match_matrix dimensions for broadcasting
            # query_mask_expanded: [batch_size, query_seq_len, 1]
            # doc_mask_expanded: [batch_size, 1, doc_seq_len]
            q_mask_exp = query_mask.unsqueeze(2)
            d_mask_exp = doc_mask.unsqueeze(1)
            
            # Combined mask: [batch_size, query_seq_len, doc_seq_len]
            # Entries are 1 if both query and doc tokens are non-padded, 0 otherwise.
            combined_mask = q_mask_exp * d_mask_exp
            
            # Apply mask: set similarity to a very low number (e.g., -1e9 or 0) where mask is 0
            # This ensures padded tokens don't contribute to kernel scores
            # Using -1 as cosine similarity ranges from -1 to 1. Padded entries won't strongly activate kernels.
            match_matrix = match_matrix * combined_mask + (1 - combined_mask) * (-1.0) 
        elif query_mask is not None:
            q_mask_exp = query_mask.unsqueeze(2)
            match_matrix = match_matrix * q_mask_exp + (1 - q_mask_exp) * (-1.0)
        elif doc_mask is not None:
            d_mask_exp = doc_mask.unsqueeze(1)
            match_matrix = match_matrix * d_mask_exp + (1 - d_mask_exp) * (-1.0)
            
        return match_matrix

    def forward(self, query_ids, doc_ids, query_mask=None, doc_mask=None):
        '''
        Forward pass of the KNRM model.

        Args:
            query_ids (torch.Tensor): Tensor of query token IDs. 
                                      Shape: [batch_size, query_seq_len]
            doc_ids (torch.Tensor): Tensor of document token IDs. 
                                    Shape: [batch_size, doc_seq_len]
            query_mask (torch.Tensor, optional): Mask for query padding. 
                                               Shape: [batch_size, query_seq_len]
            doc_mask (torch.Tensor, optional): Mask for document padding. 
                                             Shape: [batch_size, doc_seq_len]

        Returns:
            torch.Tensor: Relevance score for each query-document pair. 
                          Shape: [batch_size, 1]
        '''
        # 1. Embedding Layer
        query_embeddings = self.embedding(query_ids)
        doc_embeddings = self.embedding(doc_ids)

        # 2. Match Matrix Construction
        # match_matrix shape: [batch_size, query_seq_len, doc_seq_len]
        match_matrix = self._compute_match_matrix(query_embeddings, doc_embeddings, query_mask, doc_mask)

        # Prepare for kernel processing: expand match_matrix for broadcasting with kernels
        # mm_expanded shape: [batch_size, query_seq_len, doc_seq_len, 1]
        mm_expanded = match_matrix.unsqueeze(-1)

        # 3. Kernel Pooling
        # RBF kernel application: exp(- (x - mu)^2 / (2 * sigma^2))
        # self.mu and self.sigma shapes: [1, 1, 1, n_kernels]
        # kernel_activations shape: [batch_size, query_seq_len, doc_seq_len, n_kernels]
        kernel_activations = torch.exp(-torch.pow(mm_expanded - self.mu, 2) / (2 * torch.pow(self.sigma, 2)))

        # Mask kernel activations based on document padding before summing over doc dimension
        if doc_mask is not None:
            # doc_mask_expanded shape: [batch_size, 1, doc_seq_len, 1]
            doc_mask_expanded_for_kernels = doc_mask.unsqueeze(1).unsqueeze(-1).float()
            kernel_activations = kernel_activations * doc_mask_expanded_for_kernels

        # Sum over document dimension (soft-TF for each query term per kernel)
        # soft_tf_q shape: [batch_size, query_seq_len, n_kernels]
        soft_tf_q = torch.sum(kernel_activations, dim=2)

        # Log scaling (robust version: log(1 + x))
        # log_soft_tf_q shape: [batch_size, query_seq_len, n_kernels]
        log_soft_tf_q = torch.log(torch.clamp(soft_tf_q, min=1e-6) + 1.0) # Clamp to avoid log(0)
        # Alternative from paper: torch.log(torch.clamp(soft_tf_q, min=1e-6)) if counts are guaranteed > 0

        # Mask log_soft_tf_q based on query padding before summing over query dimension
        if query_mask is not None:
            # query_mask_expanded shape: [batch_size, query_seq_len, 1]
            query_mask_expanded_for_kernels = query_mask.unsqueeze(-1).float()
            log_soft_tf_q = log_soft_tf_q * query_mask_expanded_for_kernels

        # 4. Aggregation over Query Terms
        # Sum over query dimension (aggregate kernel features)
        # kernel_features shape: [batch_size, n_kernels]
        kernel_features = torch.sum(log_soft_tf_q, dim=1)

        # 5. Final Fully Connected (FC) Layer
        # score shape: [batch_size, 1]
        score = self.dense_layer(kernel_features)

        return score