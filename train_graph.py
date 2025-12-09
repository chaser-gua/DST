import torch
import argparse
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
import math
import torch.nn as nn
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from model import RoadModel
from utils import setup_seed, get_logger, weight_init, next_batch_index, get_test_features, process_string
from preprocess import get_road_feature, get_edge_attr, get_hyper_structure, get_w_by_hop, get_dynamic_volumn, get_dynamic_speed
from config import model_config

from collections import defaultdict
from dhg import Hypergraph


## graph contrasitive loss
def jsd_loss(z1, z2, pos_mask):
  neg_mask = 1 - pos_mask

  sim_mat = torch.mm(z1, z2.t())
      
  E_pos = math.log(2.) - F.softplus(-sim_mat)
  E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.)
  
  return (E_neg * neg_mask).sum() / neg_mask.sum() - (E_pos * pos_mask).sum() / pos_mask.sum()  


def info_nce_loss(z1, z2, pos_mask, tau=0.3, normalize=True):
    if normalize:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

    sim_mat = torch.mm(z1, z2.t()) 
    

    sim_mat = sim_mat / tau

    sim_mat_exp = torch.exp(sim_mat)
    
    pos_sim_sum = (sim_mat_exp * pos_mask).sum(dim=1)  
 
    all_sim_sum = sim_mat_exp.sum(dim=1) 

    log_prob = torch.log(pos_sim_sum / all_sim_sum)

    loss = -log_prob.mean()
    
    return loss


def ntx_loss(z1, z2, pos_mask, tau=0.5, normalize=True):
  if normalize:
      z1 = F.normalize(z1)
      z2 = F.normalize(z2)
  sim_mat = torch.mm(z1, z2.t())
  sim_mat = torch.exp(sim_mat / tau)
  
  return -torch.log((sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)).mean()  




def loss_f(z1, z2, pos_mask, measure):
  if measure == 'jsd':
    loss = jsd_loss(z1, z2, pos_mask)
  elif measure == "nce":
    loss = info_nce_loss(z1, z2, pos_mask)
  elif measure == "ntx":
    loss = ntx_loss(z1, z2, pos_mask)
    
  return loss
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--dataset", type=str, default="Porto_Taxi")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--train", type=int, default=0)
  parser.add_argument("--test", action="store_true")
  parser.add_argument("--nor", action="store_true")
  parser.add_argument("--loss", type=str, default='jsd')
  parser.add_argument("--ver", type=str, required=True)
  parser.add_argument("--best_volumn", type=int, default=0)
  parser.add_argument("--best_graph", type=int, default=0)
  parser.add_argument("--magicP", type=int, default=512)
  parser.add_argument("--hg_type", type=int, default=4)
  

  args = parser.parse_args()
  
  dataset = args.dataset
  seed = args.seed
  device = args.device
  measure = args.loss
  ver = args.ver
  train = args.train
  normalize = args.nor
  isTest = args.test
  magicP = args.magicP
  hg_type = args.hg_type
  
  
  log_path = ""
  logging = get_logger(log_dir=log_path, ver=ver, isTest=isTest)
  setup_seed(seed=seed)
  
  logging.info(f"dataset: {dataset}, seed: {seed}, device: {device}, measure: {measure}, ver: {ver}, train: {train}, hg_type: {hg_type}, normalize: {normalize}, test: {isTest}, magicP: {magicP}")
  
  geo_path = f"./data/{dataset}/roadmap.geo"
  rel_path = f"./data/{dataset}/roadmap.rel"
  traj_path = f"./data/{dataset}/traj.csv"
  
  feature_path = f"./data/{dataset}/road_feature.csv"
  if not os.path.exists(feature_path):
    road_features = get_road_feature(geo_path=geo_path, dataset=dataset, save_path=feature_path)
  else:
    road_features = pd.read_csv(feature_path)
  logging.info(f"==Road_features is ready")  
  
  num_nodes = len(road_features)  
  
  # Embedding config
  model_config["road_id_in"] = road_features["road_id"].max() + 1
  model_config["lon_in"] = road_features["lon_grid"].max() + 1
  model_config["lat_in"] = road_features["lat_grid"].max() + 1
  model_config["highway_in"] = road_features["highway_info"].max() + 1
  model_config["lanes_in"] = road_features["lanes_info"].max() + 1
  
  model_config["length_in"] = road_features["length_info"].max() + 1
  model_config["maxspeed_in"] = road_features["maxspeed_info"].max() + 1
  

  raw_feature_dis = torch.LongTensor(road_features.iloc[:, :].astype(int).values).to(device)
  raw_feature_con = torch.FloatTensor(road_features.iloc[:, -5:-2].astype(float).values).to(device)
  
  # edge_attr
  edge_attr_path = f"./data/{dataset}/edge_attr.csv"
  if not os.path.exists(edge_attr_path):
    edge_attr = get_edge_attr(geo_path=geo_path, rel_path=rel_path, save_path=edge_attr_path)
  else:
    edge_attr = pd.read_csv(edge_attr_path)
    
  edge_index = get_edge_attr(geo_path=geo_path, rel_path=rel_path, save_path=edge_attr_path, only_index=True)
  edge_index = torch.LongTensor(edge_index).to(device)
  edge_attr = torch.FloatTensor(edge_attr.values).to(device)
  # edge_attr = torch.tensor(edge_attr.values, dtype=torch.float16).to(device)
  
  logging.info("==edge_ettr is ready==")
  
  # hypergraph structure
  region_path = f"./data/{dataset}/seg2region.csv"

  seg2region = pd.read_csv(region_path)
  
  hg_stru = Hypergraph(len(seg2region), None)

  region_group = seg2region.groupby("region_id")
  
  num_hyperedge = 0
  # type_1, type_2, type_3 = 0,0,0
  type_1, type_2, type_3, type_4 = 0,0,0,0
  
  hyperedge_type = hg_type
  
  if hyperedge_type >= 1:
    for name, group in tqdm(region_group, total=len(region_group)):
      hg_stru.add_hyperedges(e_list=group["geo_id"].to_list(), merge_op="sum")
      num_hyperedge += 1
      type_1 += 1
      
  print(f"1-num_hyperedge: {type_1}; total-num_hyperedge: {num_hyperedge}")
    
  if hyperedge_type >= 2:
    # hg_2 = Hypergraph(len(seg2region), None)
    long_range_info = road_features["highway_info"]
    max_number = long_range_info.max()+1
    for i in tqdm(range(max_number)):
      hg_stru.add_hyperedges(np.where(long_range_info == i)[0].tolist(), merge_op="sum")
      num_hyperedge += 1
      type_2 += 1
    
  print(f"2-num_hyperedge: {type_2}; total-num_hyperedge: {num_hyperedge}")
  
  if hyperedge_type >= 3:
    # hg_3 = Hypergraph(len(seg2region), None)
    rel_file = pd.read_csv(rel_path)
    origin = rel_file['origin_id'].to_list()
    destination = rel_file['destination_id'].to_list()
    for i in tqdm(range(len(origin))):
      if origin[i] not in destination:
        hg_stru.add_hyperedges((origin[i], destination[i]))
        num_hyperedge += 1
        type_3 += 1
        
  print(f"3-num_hyperedge: {type_3}; total-num_hyperedge: {num_hyperedge}")       
      
  print(f"4-num_hyperedge: {type_4}; total-num_hyperedge: {num_hyperedge}")  

  logging.info("==hg_structure is ready==")
  
  # weight
  weight_path = f"./data/{dataset}/p_hop.npy"
  if not os.path.exists(weight_path):
    hop_weight = get_w_by_hop(geo_path=geo_path, traj_path=traj_path, save_path=weight_path)
  else:
    hop_weight = np.load(weight_path)
  
    
  for i in tqdm(range(len(hop_weight))):
    row_sum = sum(hop_weight[i])
    if row_sum == 0:
      hop_weight[i, i] = 1
    else:
      for j in range(len(hop_weight)):
        hop_weight[i, j] = hop_weight[i, j] / row_sum
        
  
  hop_weight = torch.FloatTensor(hop_weight).to(device)
  
  
  logging.info("==hop weight is ready==")
  
  
  """
  #######################
  ######Train graph######
  #######################
  """

  learning_rate = model_config["learning_rate"]
  weight_decay = model_config["weight_decay"]
  max_episode = model_config["graph_epoch"]
  # max_episode = 5000
  best_graph_loss = 10e9
  best_graph_epoch = 0
  num_nodes =  len(hop_weight)
  if not isTest:
    road_model = RoadModel(model_config=model_config, w=hop_weight, n=num_nodes).to(device)
    optimizer = torch.optim.AdamW(road_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    road_model.apply(weight_init)
    
    batch_size = magicP 
    num_nodes =  len(hop_weight)
    num_batches = (num_nodes + batch_size - 1) 

    for cur_episode in range(max_episode):  
      road_model.train() 
      optimizer.zero_grad() 
      
      g_view, hg_view = road_model(raw_feature_dis, raw_feature_con, edge_index, hg_stru, edge_attr)
      
      # all_view = g_view + hg_view
      
      total_loss = 0.0
      
      for i in range(num_batches):
          start_idx = i * batch_size
          end_idx = min((i + 1) * batch_size, num_nodes)
          
          g_batch = g_view[start_idx:end_idx].to(device)
          hg_batch = hg_view[start_idx:end_idx].to(device)
          # all_batch = all_view[start_idx:end_idx].to(device)
          
          pos_mask_batch = torch.eye(end_idx - start_idx).to(device) 
          
          total_loss += loss_f(g_batch, hg_batch, pos_mask_batch, measure)

      graph_loss = total_loss / num_batches
      graph_loss.backward()
      optimizer.step()
      
      if (cur_episode+1) % 250 == 0:
        logging.info(f"Episode {cur_episode}/{max_episode}, Loss: {graph_loss.item():.4f}")
    
    logging.info("!!!Training graph end!!!")
  
  

  

