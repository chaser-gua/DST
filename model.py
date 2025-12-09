import torch
import torch.nn as nn
from torch_geometric.nn.models import GAT, GCN
import dhg
import math
import os

from torch.utils.data import DataLoader, TensorDataset
from utils import weight_init


"""static message modeling"""
# Feat
class FeatEmbedding(nn.Module):
  def __init__(self, model_config):
    super(FeatEmbedding, self).__init__()
    
    id_out = model_config["emb_out"]["id"]
    dis_out = model_config["emb_out"]["dis"]
    con_out = model_config["emb_out"]["con"]
    
    # discrete
    self.emb_id = nn.Embedding(model_config["road_id_in"], id_out)
    self.emb_lon = nn.Embedding(model_config["lon_in"], dis_out)
    self.emb_lat = nn.Embedding(model_config["lat_in"], dis_out)
    self.emb_highway = nn.Embedding(model_config["highway_in"], dis_out)
    self.emb_oneway = nn.Embedding(model_config["oneway_in"], dis_out)
    self.emb_tnunel = nn.Embedding(model_config["tunnel_in"], dis_out)
    self.emb_bridge = nn.Embedding(model_config["bridge_in"], dis_out)
    self.emb_lanes = nn.Embedding(model_config["lanes_in"], dis_out)
    
    self.emb_length = nn.Embedding(model_config["length_in"], dis_out)
    self.emb_maxspeed = nn.Embedding(model_config["maxspeed_in"], dis_out)
    

  def forward(self, raw_feature_dis, raw_feature_con):
    
    return torch.cat((
      self.emb_id(raw_feature_dis[:, 0]),
      self.emb_lon(raw_feature_dis[: , 1]),
      self.emb_lat(raw_feature_dis[: , 2]),
      self.emb_tnunel(raw_feature_dis[: , 5]),
      self.emb_bridge(raw_feature_dis[: , 6]),
      self.emb_lanes(raw_feature_dis[: , 7]),
      self.emb_length(raw_feature_dis[: , 8]),
      self.emb_maxspeed(raw_feature_dis[: , 9]),
      self.emb_highway(raw_feature_dis[: , 3]),
      self.emb_oneway(raw_feature_dis[: , 4]),
      ), dim = 1)

# pos
class PositionalEncoding(nn.Module):
  def __init__(self, dim, dropout=0.1, max_len=30000):
    super(PositionalEncoding, self).__init__()
    self.dropout = nn.Dropout(p=dropout)

    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    pe = pe.unsqueeze(0).transpose(0, 1) #.to(device)

    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[:x.size(0), :]
    return self.dropout(x)


"""from https://github.com/iMoonLab/DeepHypergraph/blob/0.9.4/dhg/nn/convs/hypergraphs/hgnnp_conv.py"""
class HGNNPConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        use_bn: bool = False,
        drop_rate: float = 0.5,
        is_last: bool = False,
    ):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, X, hg) -> torch.Tensor:
        X = self.theta(X)
        X = hg.v2v(X, aggr="mean") 
        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X
      

class HGNNP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hid_channels: int,
        num_classes: int,
        use_bn: bool = False,
        drop_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNPConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate)
        )
        self.layers.append(
            HGNNPConv(hid_channels, num_classes, use_bn=use_bn, is_last=True)
        )

    def forward(self, X, hg) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X, hg)
        return X  


class RoadModel(nn.Module):
  def __init__(self, model_config, w, n):
    super(RoadModel, self).__init__()
    
    self.w = nn.Parameter(w, requires_grad=True)

    
    # Embedding
    self.emb = FeatEmbedding(model_config=model_config)
  
    """"""
    # GNN
    in_size = model_config["emb_out"]['dis'] * 9 
    
    self.gat = GAT(
        in_channels=in_size,
        hidden_channels=model_config['hid_size'],
        num_layers=2,
        out_channels=model_config['hid_size'],
        dropout=model_config['drop_rate'],
    )
  
    self.hgnn = HGNNP(in_channels=in_size,
                      hid_channels=model_config['hid_size'],
                      num_classes=(model_config["hid_size"]),
                      drop_rate=model_config['drop_rate'])
    
    
    
  def forward(self, x_dis, x_con, edge_index, hg_stru, edge_attr):
    x = self.emb(x_dis, x_con)
    x_p = torch.mm(self.w, x)
    g_view = self.gat(x_p, edge_index, edge_attr)
    hg_view = self.hgnn(x_p, hg_stru)
    
    return g_view, hg_view

  
