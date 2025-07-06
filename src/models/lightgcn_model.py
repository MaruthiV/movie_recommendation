import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    This implementation uses regular PyTorch operations instead of torch_sparse
    for better compatibility and easier installation.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        num_layers: int = 3,
        device: str = 'cpu'
    ):
        """
        Initialize LightGCN model.
        
        Args:
            num_users: Number of users in the dataset
            num_items: Number of items in the dataset
            embedding_dim: Dimension of user and item embeddings
            num_layers: Number of LightGCN layers
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super(LightGCN, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.device = device
        
        # Initialize user and item embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings with Xavier uniform distribution
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        self.to(device)
    
    def _create_adjacency_matrix(self, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Create normalized adjacency matrix from edge index.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            
        Returns:
            Normalized adjacency matrix
        """
        # Create sparse adjacency matrix
        num_nodes = self.num_users + self.num_items
        adj_matrix = torch.zeros((num_nodes, num_nodes), device=self.device)
        
        # Add edges to adjacency matrix
        adj_matrix[edge_index[0], edge_index[1]] = 1.0
        adj_matrix[edge_index[1], edge_index[0]] = 1.0  # Make it symmetric
        
        # Calculate degree matrix
        degree = adj_matrix.sum(dim=1, keepdim=True)
        degree = torch.clamp(degree, min=1.0)  # Avoid division by zero
        
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        degree_sqrt = torch.sqrt(degree)
        adj_normalized = adj_matrix / degree_sqrt / degree_sqrt.t()
        
        return adj_normalized
    
    def forward(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LightGCN.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            
        Returns:
            Tuple of (user_embeddings, item_embeddings)
        """
        # Create normalized adjacency matrix
        adj_matrix = self._create_adjacency_matrix(edge_index)
        
        # Initialize embeddings
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Store embeddings from each layer
        emb_list = [all_emb]
        
        # LightGCN layers
        for _ in range(self.num_layers):
            # Graph convolution: E^(l+1) = D^(-1/2) * A * D^(-1/2) * E^(l)
            all_emb = torch.mm(adj_matrix, all_emb)
            emb_list.append(all_emb)
        
        # Average embeddings from all layers
        all_emb = torch.stack(emb_list, dim=1).mean(dim=1)
        
        # Split back into user and item embeddings
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items], dim=0)
        
        return user_emb, item_emb
    
    def predict(self, user_indices: torch.Tensor, item_indices: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict interaction scores between users and items.
        
        Args:
            user_indices: Tensor of user indices
            item_indices: Tensor of item indices
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            
        Returns:
            Tensor of interaction scores
        """
        # Get embeddings
        user_emb, item_emb = self.forward(edge_index)
        
        # Get embeddings for specific users and items
        user_emb_selected = user_emb[user_indices]
        item_emb_selected = item_emb[item_indices]
        
        # Calculate interaction scores (dot product)
        scores = torch.sum(user_emb_selected * item_emb_selected, dim=1)
        
        return scores
    
    def bpr_loss(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor,
                 edge_index: torch.Tensor, reg_weight: float = 1e-4) -> torch.Tensor:
        """
        Calculate BPR (Bayesian Personalized Ranking) loss.
        
        Args:
            users: Tensor of user indices
            pos_items: Tensor of positive item indices
            neg_items: Tensor of negative item indices
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            reg_weight: Weight for L2 regularization
            
        Returns:
            BPR loss tensor
        """
        # Get embeddings
        user_emb, item_emb = self.forward(edge_index)
        
        # Get embeddings for users, positive items, and negative items
        user_emb_selected = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        
        # Calculate scores
        pos_scores = torch.sum(user_emb_selected * pos_item_emb, dim=1)
        neg_scores = torch.sum(user_emb_selected * neg_item_emb, dim=1)
        
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        # L2 regularization
        if reg_weight > 0:
            l2_reg = torch.norm(user_emb_selected, p=2) + torch.norm(pos_item_emb, p=2) + torch.norm(neg_item_emb, p=2)
            bpr_loss += reg_weight * l2_reg
        
        return bpr_loss
    
    def get_user_item_embeddings(self, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get user and item embeddings for inference.
        
        Args:
            edge_index: Tensor of shape (2, num_edges) containing user-item interactions
            
        Returns:
            Tuple of (user_embeddings, item_embeddings) without gradients
        """
        with torch.no_grad():
            user_emb, item_emb = self.forward(edge_index)
        return user_emb, item_emb 