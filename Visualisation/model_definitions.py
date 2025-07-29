import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import scatter
import torch
import torch.nn as nn

class GGNN(nn.Module):
  def __init__(self, num_node_features=306, hidden_channels=300, num_edge_types=3, num_iterations=3):
      super(GGNN, self).__init__()
      self.num_iterations = num_iterations
      self.hidden_channels = hidden_channels
      self.lin_init = nn.Linear(num_node_features, hidden_channels)
      self.edge_weights = nn.ModuleList([
          nn.Linear(hidden_channels, hidden_channels) for _ in range(num_edge_types)
      ])
      self.bias = nn.Parameter(torch.zeros(hidden_channels))
      self.gru = nn.GRUCell(hidden_channels, hidden_channels)
      self.attn_fi = nn.Linear(hidden_channels + num_node_features, hidden_channels)
      self.attn_fj = nn.Linear(hidden_channels + num_node_features, hidden_channels)
      self.attn_score = nn.Linear(hidden_channels, 1)
      self.fc = nn.Linear(hidden_channels, hidden_channels)
  
  def forward(self, data):
      x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
      batch = data.batch if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
      
      if edge_index.max().item() >= x.size(0):
          raise ValueError(f"edge_index contains invalid indices: max {edge_index.max().item()}, num_nodes {x.size(0)}")
      
      h = self.lin_init(x)
      
      for _ in range(self.num_iterations):
          messages = torch.zeros_like(h)
          for edge_type in range(len(self.edge_weights)):
              mask = edge_attr == edge_type
              if mask.sum() > 0:
                  edge_index_type = edge_index[:, mask]
                  source_nodes = edge_index_type[0]
                  target_nodes = edge_index_type[1]
                  message = self.edge_weights[edge_type](h[source_nodes]) + self.bias
                  messages.index_add_(0, target_nodes, message)
          h = self.gru(messages, h)
      
      concat = torch.cat([h, x], dim=1)
      fi = self.attn_fi(concat)
      scores = torch.sigmoid(self.attn_score(fi))
      fj = torch.tanh(self.attn_fj(concat))
      weighted = scores * fj
      h_g = scatter(weighted, batch, dim=0, reduce='sum')
      h_g = torch.tanh(h_g)
      
      data.attn_scores = scores
      h_g = self.fc(h_g)
      return h_g
  
  


class DeepSVDDLoss(nn.Module):
  def __init__(self, c, mu=0.05, lambda_reg=0.001):
      super(DeepSVDDLoss, self).__init__()
      self.c = nn.Parameter(c, requires_grad=False)
      self.R = nn.Parameter(torch.tensor(1.0))
      self.mu = mu
      self.lambda_reg = lambda_reg
  
  def forward(self, embeddings, model):
      dist = torch.sum((embeddings - self.c) ** 2, dim=1)
      violation = torch.clamp(dist - self.R ** 2, min=0).mean()
      loss = self.R ** 2 + (1 / self.mu) * violation
      reg = sum(torch.norm(param, p='fro') ** 2 for param in model.parameters())
      loss += (self.lambda_reg / 2) * reg
      return loss