import unittest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from lightgcn_model import LightGCN


class TestLightGCN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.num_users = 100
        self.num_items = 200
        self.embedding_dim = 64
        self.num_layers = 3
        self.device = 'cpu'
        
        # Create sample edge index (user-item interactions)
        self.edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2, 3, 3],  # users
            [0, 1, 1, 2, 2, 3, 3, 4]   # items (offset by num_users)
        ], dtype=torch.long)
        
        # Adjust item indices to account for bipartite graph
        self.edge_index[1] += self.num_users
        
        self.model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            device=self.device
        )

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.num_users, self.num_users)
        self.assertEqual(self.model.num_items, self.num_items)
        self.assertEqual(self.model.embedding_dim, self.embedding_dim)
        self.assertEqual(self.model.num_layers, self.num_layers)
        self.assertEqual(self.model.device, self.device)
        
        # Check embedding dimensions
        self.assertEqual(self.model.user_embedding.weight.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(self.model.item_embedding.weight.shape, (self.num_items, self.embedding_dim))
        
        # Check that embeddings are initialized (not all zeros)
        self.assertFalse(torch.allclose(self.model.user_embedding.weight, torch.zeros_like(self.model.user_embedding.weight)))
        self.assertFalse(torch.allclose(self.model.item_embedding.weight, torch.zeros_like(self.model.item_embedding.weight)))

    def test_forward_pass(self):
        """Test forward pass of the model."""
        user_emb, item_emb = self.model.forward(self.edge_index)
        
        # Check output shapes
        self.assertEqual(user_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, self.embedding_dim))
        
        # Check that embeddings are not all zeros
        self.assertFalse(torch.allclose(user_emb, torch.zeros_like(user_emb)))
        self.assertFalse(torch.allclose(item_emb, torch.zeros_like(item_emb)))

    def test_predict(self):
        """Test prediction method."""
        user_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        item_indices = torch.tensor([0, 1, 2], dtype=torch.long)
        
        scores = self.model.predict(user_indices, item_indices, self.edge_index)
        
        # Check output shape
        self.assertEqual(scores.shape, (3,))
        
        # Check that scores are not all zeros
        self.assertFalse(torch.allclose(scores, torch.zeros_like(scores)))

    def test_bpr_loss(self):
        """Test BPR loss computation."""
        users = torch.tensor([0, 1, 2], dtype=torch.long)
        pos_items = torch.tensor([0, 1, 2], dtype=torch.long)
        neg_items = torch.tensor([3, 4, 5], dtype=torch.long)
        
        loss = self.model.bpr_loss(users, pos_items, neg_items, self.edge_index)
        
        # Check that loss is a scalar tensor
        self.assertEqual(loss.shape, ())
        
        # Check that loss is positive (BPR loss should be positive)
        self.assertGreater(loss.item(), 0)

    def test_bpr_loss_with_regularization(self):
        """Test BPR loss with regularization."""
        users = torch.tensor([0, 1, 2], dtype=torch.long)
        pos_items = torch.tensor([0, 1, 2], dtype=torch.long)
        neg_items = torch.tensor([3, 4, 5], dtype=torch.long)
        
        # Test with different regularization weights
        loss_no_reg = self.model.bpr_loss(users, pos_items, neg_items, self.edge_index, reg_weight=0.0)
        loss_with_reg = self.model.bpr_loss(users, pos_items, neg_items, self.edge_index, reg_weight=1e-4)
        
        # Loss with regularization should be greater than without
        self.assertGreater(loss_with_reg.item(), loss_no_reg.item())

    def test_get_user_item_embeddings(self):
        """Test getting user and item embeddings."""
        user_emb, item_emb = self.model.get_user_item_embeddings(self.edge_index)
        
        # Check output shapes
        self.assertEqual(user_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, self.embedding_dim))
        
        # Check that embeddings are detached (no gradients)
        self.assertFalse(user_emb.requires_grad)
        self.assertFalse(item_emb.requires_grad)

    def test_different_embedding_dimensions(self):
        """Test model with different embedding dimensions."""
        embedding_dim = 128
        model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=embedding_dim,
            num_layers=self.num_layers,
            device=self.device
        )
        
        user_emb, item_emb = model.forward(self.edge_index)
        self.assertEqual(user_emb.shape, (self.num_users, embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, embedding_dim))

    def test_different_num_layers(self):
        """Test model with different numbers of layers."""
        num_layers = 1
        model = LightGCN(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            num_layers=num_layers,
            device=self.device
        )
        
        user_emb, item_emb = model.forward(self.edge_index)
        self.assertEqual(user_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, self.embedding_dim))

    def test_empty_edge_index(self):
        """Test model with empty edge index."""
        empty_edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # Should not raise an error
        user_emb, item_emb = self.model.forward(empty_edge_index)
        self.assertEqual(user_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, self.embedding_dim))

    def test_single_interaction(self):
        """Test model with single user-item interaction."""
        single_edge_index = torch.tensor([[0], [self.num_users]], dtype=torch.long)
        
        user_emb, item_emb = self.model.forward(single_edge_index)
        self.assertEqual(user_emb.shape, (self.num_users, self.embedding_dim))
        self.assertEqual(item_emb.shape, (self.num_items, self.embedding_dim))

    def test_model_device_moving(self):
        """Test moving model to different device."""
        if torch.cuda.is_available():
            device = 'cuda'
            model = LightGCN(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=self.embedding_dim,
                num_layers=self.num_layers,
                device=device
            )
            
            # Check that model is on correct device
            self.assertEqual(next(model.parameters()).device.type, device)
            
            # Test forward pass on GPU
            edge_index_gpu = self.edge_index.to(device)
            user_emb, item_emb = model.forward(edge_index_gpu)
            self.assertEqual(user_emb.device.type, device)
            self.assertEqual(item_emb.device.type, device)

    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        self.model.train()
        
        # Create a simple loss
        user_emb, item_emb = self.model.forward(self.edge_index)
        loss = user_emb.sum() + item_emb.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(self.model.user_embedding.weight.grad)
        self.assertIsNotNone(self.model.item_embedding.weight.grad)

    def test_model_parameters(self):
        """Test that model has the expected number of parameters."""
        total_params = sum(p.numel() for p in self.model.parameters())
        expected_params = self.num_users * self.embedding_dim + self.num_items * self.embedding_dim
        
        self.assertEqual(total_params, expected_params)

    def test_model_training_mode(self):
        """Test model in training and evaluation modes."""
        # Training mode
        self.model.train()
        self.assertTrue(self.model.training)
        
        # Evaluation mode
        self.model.eval()
        self.assertFalse(self.model.training)

    def test_invalid_inputs(self):
        """Test model behavior with invalid inputs."""
        # Test with user indices out of range
        with self.assertRaises(IndexError):
            invalid_users = torch.tensor([self.num_users + 1], dtype=torch.long)
            invalid_items = torch.tensor([0], dtype=torch.long)
            self.model.predict(invalid_users, invalid_items, self.edge_index)
        
        # Test with item indices out of range
        with self.assertRaises(IndexError):
            valid_users = torch.tensor([0], dtype=torch.long)
            invalid_items = torch.tensor([self.num_items + 1], dtype=torch.long)
            self.model.predict(valid_users, invalid_items, self.edge_index)


if __name__ == '__main__':
    unittest.main() 