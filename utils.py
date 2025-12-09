import pickle
import numpy as np
from scipy import sparse
import random
import copy
import logging

from datetime import datetime
import os
import sys
import torch

import pandas as pd
from tqdm import tqdm
import math

import torch.nn.init as init
import torch.nn as nn


class Dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__


def dict_to_object(dictObj):
  if not isinstance(dictObj, dict):
    return dictObj
  inst=Dict()
  for k,v in dictObj.items():
    inst[k] = dict_to_object(v)
  return inst

def setup_seed(seed):
  torch.manual_seed(seed)  
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True

def get_logger(log_dir, ver, isTest):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f'{formatted_time}_{ver}_{"test" if isTest else "train"}.log')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
  

def to_cartesian(lat, lon, R=6371):
    lat = np.radians(lat)
    lon = np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])

def average_direction_vector(coords):
    vectors = []
    for i in range(len(coords.coords) - 1):
        start = to_cartesian(coords.coords[i][0], coords.coords[i][1])
        end = to_cartesian(coords.coords[i+1][0], coords.coords[i+1][1])
        vectors.append(end - start)
    average_vector = np.mean(vectors, axis=0)
    return average_vector
  
def normalize_angle(angle):
    angle = angle % (2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return angle

def angle_between_lines(coords1, coords2):
    v1 = average_direction_vector(coords1)
    v2 = average_direction_vector(coords2)

    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
      return 0 

    cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_radians = np.arccos(cos_theta)
    
    normalized_angle = normalize_angle(angle_radians)

    return normalized_angle
  

def next_batch_index(ds, bs, shuffle=True):
    num_batches = math.ceil(ds / bs)

    index = np.arange(ds)
    if shuffle:
        index = np.random.permutation(index)

    for i in range(num_batches):
        if i == num_batches - 1:
            batch_index = index[bs * i:]
        else:
            batch_index = index[bs * i: bs * (i + 1)]
        yield batch_index
  
  
def process_string(s):
    if pd.isna(s):  
        return None
    categories = s.split(',') 
    categories = sorted(set(categories))  
    return ','.join(categories)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.TransformerEncoderLayer):
        # print("==in_Transformer==")

        init.xavier_normal_(m.linear1.weight.data)
        init.xavier_normal_(m.linear2.weight.data)
        if m.linear1.bias is not None:
            init.constant_(m.linear1.bias.data, 0)
        if m.linear2.bias is not None:
            init.constant_(m.linear2.bias.data, 0)    
    elif isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        # init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5 / m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)
    elif isinstance(m, nn.Bilinear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            
            
def get_test_features(geo_path, traj_path, save_path):
    
    geo_file = pd.read_csv(geo_path)

    for index, row in tqdm(geo_file.iterrows(), total = len(geo_file)):
        oneway = row['oneway'] if not pd.isna(row['oneway']) else False
        lanes = row['lanes'] if not pd.isna(row['lanes']) else '0'
        highway = row['highway'] if not pd.isna(row['highway']) else 'unclassified'
        length = row['length'] if not pd.isna(row['length']) else 0
        bridge = 1 if not pd.isna(row['bridge']) else 0
        tunnel = 1 if not pd.isna(row['tunnel']) else 0
        features.append((index, oneway, lanes, highway, length, bridge, tunnel))

    feature_df = pd.DataFrame(features, columns=['road_id', 'oneway', 'lanes', 'highway', 'length', 'bridge', 'tunnel'])
    feature_df['oneway'] = feature_df['oneway'].map(lambda x: int(x) if type(x) != list else 0)
    feature_df['lanes'] = feature_df['lanes'].map(lambda x: eval(x[0]) if type(x) == list else x)
    feature_df['highway'] = feature_df['highway'].map(lambda x: x[0] if type(x) == list else x)
    tmp_feat = feature_df.loc[feature_df['highway'] != 'unclassified']['highway']
    highway2idx = {hw: i + 1 for i, hw in enumerate(tmp_feat.value_counts().index)}
    highway2idx['unclassified'] = 0
    feature_df['highway_id'] = feature_df['highway'].map(highway2idx)
    feature_df['length_id'] = feature_df['length'].map(lambda x: math.ceil(x / 100))
    feature_df.sort_values('road_id').reset_index(inplace=True)
    
    traj = pd.read_csv(traj_path)

    st = []
    pt = []
    tt = []

    time_str = "%Y-%m-%dT%H:%M:%SZ"
    
    for index, row in tqdm(traj.iterrows(), total = len(traj)):
    # try:
        time = row['time_list'].split(',')
        times = [datetime.strptime(_, time_str) for _ in time]
        st.append(times[0])
        tt.append((times[-1]-times[0]).total_seconds())
        pt.append([(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])])
    # except:
        # print(index)
        
    traj['st'] = st
    traj['pt'] = pt
    traj['tt'] = tt
    
    speed_dict = {}
    rs_length = feature_df['length']
    for _, row in tqdm(traj.iterrows(), total = len(traj)):
        path = eval(row['rid_list'])
        pt = row['pt']
        # print(pt)
        for i, rs in enumerate(path):
            if i != len(path) - 1 and pt[i] > 0 :
                if pt[i] > 0:
                    tmp = rs_length[rs] / pt[i]
                    if tmp < 30:
                        avg, n = speed_dict.get(rs, (0, 0))
                        speed_dict[rs] = ((avg * n + tmp) / (n + 1), (n + 1))
                        
    feature_df['road_speed'] = [speed_dict.get(i, (0, 0))[0] for i in range(len(feature_df))]
    
    speed_dict = {}
    rs_length = feature_df['length']
    # data_df = pd.read_csv(os.path.join(data_path, file_name))
    for _, row in tqdm(traj.iterrows(), total = len(traj)):
        path = eval(row['rid_list'])
        tl = sum([rs_length[rs] for rs in path])
        tmp = tl / row['tt']
        for i, rs in enumerate(path):
            avg, n = speed_dict.get(rs, (0, 0))
            speed_dict[rs] = ((avg * n + tmp) / (n + 1), (n + 1))
            
    feature_df['traj_speed'] = [speed_dict.get(i, (0, 0))[0] for i in range(len(feature_df))]
    
    feature_df.to_csv(save_path, index=False)
