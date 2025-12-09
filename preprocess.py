import pandas as pd
import numpy as np
from tqdm import tqdm
from shapely.geometry import LineString
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from haversine import haversine, haversine_vector, Unit
from dhg import Hypergraph
import holidays
from datetime import datetime
import csv
from ast import literal_eval

from dataset import MapManager
from utils import angle_between_lines, process_string



def get_road_feature(geo_path, dataset, save_path):
  """
  np.array
  """

  road_feature = {}
  
  geo_file = pd.read_csv(geo_path)
  
  # cooridinates
  lon_grid = []
  lat_grid = []
  map_manager = MapManager(dataset)
  
  for i in tqdm(range(len(geo_file))):
    coordinates = eval(geo_file.loc[i, 'coordinates'])
    road_line = LineString(coordinates=coordinates)
    center_coord = road_line.centroid
    x, y = map_manager.gps2grid(center_coord.x, center_coord.y)
    lon_grid.append(x)
    lat_grid.append(y)
    
  lon_grid = np.array(lon_grid)
  lat_grid = np.array(lat_grid)

  # highway
  highway_info = geo_file[['highway']].fillna('unclassified')
  highway_info = geo_file['highway'].values.tolist()
  if dataset in ['BJ_Taxi', 'SF_Taxi']:
      for i in range(len(highway_info)):
          if highway_info[i].startswith('[') and highway_info[i].endswith(']'):
              info = eval(highway_info[i])
              highway_info[i] = info[0] if info[0] != 'unclassified' else info[1]
  le = LabelEncoder()
  highway_info = le.fit_transform(highway_info)
  
  # oneway
  geo_file["oneway"] = (geo_file["oneway"].fillna(0).replace({"yes" : 1 , "True": 1, "False": 0, "[False, True]": 1}))
  oneway_info = geo_file['oneway'].to_numpy()
  
  # tunel
  geo_file["tunnel"] = (geo_file["tunnel"].fillna(0).replace(
    ["yes", "building_passage", "culvert", "['yes', 'building_passage']"], 1
    )
  )
  tunnel_info = geo_file['tunnel'].to_numpy()
  
  # bridge
  geo_file["bridge"] = (geo_file["bridge"].fillna(0).replace([
    "yes","viaduct",
    "['yes', 'viaduct']",
    "cantilever",
    "['yes', 'movable']",
    "movable",
    "['no', 'yes']",
    "['viaduct', 'yes']",
    ], 1))
  bridge_info = geo_file['bridge'].to_numpy()
  
  # lanes
  geo_file["lanes"] = geo_file["lanes"].astype(str).str.extract(r"(\d+)")
  
  ### continue ###
  
  # maxspeed
  geo_file["maxspeed"] = geo_file["maxspeed"].astype(str).str.extract(r"(\d+)")
  
  # length
  geo_file["length"] = (geo_file["length"] - geo_file["length"].min()) / (
      geo_file["length"].max() - geo_file["length"].min()
  )
  length_info = geo_file['length'].to_numpy()
  
  imputer = KNNImputer(n_neighbors=1)
  imputed = imputer.fit_transform(geo_file[['lanes', 'maxspeed']])
  
  # lanes_index = geo_file.columns.get_loc("lanes")
  # maxspeed_index = geo_file.columns.get_loc("maxspeed")
  
  geo_file["lanes"] = imputed[:, 0].astype(int)
  
  geo_file["maxspeed"] = imputed[:, 1].astype(int)
  geo_file["maxspeed"] = (geo_file["maxspeed"] - geo_file["maxspeed"].min()) / (
      geo_file["maxspeed"].max() - geo_file["maxspeed"].min()
  )
  
  lanes_info = geo_file['lanes'].to_numpy()
  maxspeed_info = geo_file['maxspeed'].to_numpy()
  
  road_feature["road_id"] = np.arange(len(geo_file))
  road_feature["lon_grid"] = lon_grid
  road_feature["lat_grid"] = lat_grid
  road_feature["highway_info"] = highway_info
  road_feature["oneway_info"] = oneway_info
  road_feature["tunnel_info"] = tunnel_info
  road_feature["bridge_info"] = bridge_info
  road_feature["lanes_info"] = lanes_info
  road_feature["length_info"] = length_info
  road_feature["maxspeed_info"] = maxspeed_info
  
  bins = [-0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
  label = range(10)

  road_feature['length_info'] = pd.cut(road_feature['length_info'], bins=bins, labels=label)
  road_feature['maxspeed_info'] = pd.cut(road_feature['maxspeed_info'], bins=bins, labels=label)
  
  road_feature = pd.DataFrame(road_feature)
  
  road_feature.to_csv(save_path, index=None)
  
  return road_feature
  
  
def get_edge_attr(geo_path, rel_path, save_path, only_index = False):
  """
  np.array
  """
  
  # edge_index
  edge_attr = {}
  
  geo_file = pd.read_csv(geo_path)
  rel_file = pd.read_csv(rel_path)
  
  edge_index = []
  src = rel_file[['origin_id']].values.T
  tgt = rel_file[['destination_id']].values.T
  
  edge_index.append(src[0])
  edge_index.append(tgt[0])
  
  edge_index = np.array(edge_index)
  
  if only_index:
    return edge_index

  # angle and distance
  angle_info = [] 
  distance_info = []
  
  for i in tqdm(range(edge_index.shape[1])):
    road_src_coo = eval(geo_file.loc[edge_index[0][i], "coordinates"])
    road_tgt_coo = eval(geo_file.loc[edge_index[1][i], "coordinates"])
    road_src = LineString(coordinates=road_src_coo)
    road_tgt = LineString(coordinates=road_tgt_coo)
    angle_info.append(angle_between_lines(road_src, road_tgt))
    distance_info.append(haversine((road_src.centroid.y, road_src.centroid.x), (road_tgt.centroid.y, road_tgt.centroid.x), unit=Unit.METERS))
    
  angle_info = np.array(angle_info)
  distance_info = np.array(distance_info)
  
  # edge_attr["edge_index"] = edge_index
  edge_attr["angle_info"] = angle_info
  edge_attr["distance_info"] = distance_info
  
  edge_attr = pd.DataFrame(edge_attr)
  edge_attr.to_csv(save_path, index=None)
  
  return edge_attr
  
    

def get_w_by_hop(geo_path, traj_path, save_path):
  geo_file= pd.read_csv(geo_path)
  traj = pd.read_csv(traj_path)
  
  weight = [[0 for i in range(len(geo_file))] for j in range(len(geo_file))]
  
  for index, row in tqdm(traj.iterrows(), total=len(traj)):
    rid_list = eval(row['rid_list'])
    traj_len = len(rid_list)
    for i in range(traj_len - 1):
      max_hop = traj_len - 1 - i
      for j in range(i+1, traj_len):
        weight[rid_list[i]][rid_list[j]] += max_hop
        max_hop -= 1
  
  weight = np.array(weight)
  
  np.save(save_path, weight)
    
  return weight


def get_dynamic_volumn(geo_path, traj_path, save_path, dataset):

  if dataset.startswith("BJ"):
    holiday = holidays.China(years=2015)
  elif dataset.startswith("Porto"):
    holiday = holidays.Portugal()
  else:
    holiday = holidays.US()

  geo_file = pd.read_csv(geo_path)[['geo_id', 'highway']]
  traj_file = pd.read_csv(traj_path)

  for i in range(1, 49):
    geo_file[f'{i}'] = 0
    
  for index, row in tqdm(traj_file.iterrows(), total=len(traj_file)):
    path = eval(row['rid_list'])
    time = row["time_list"].split(',')
    time = [datetime.strptime(_, "%Y-%m-%dT%H:%M:%SZ") for _ in time]
    for i in range(len(path)):
      which_time = time[i].hour + 1
      if time[i].weekday() >= 5 or time[i].date() in holiday:
        geo_file.loc[path[i], f"{which_time + 24}"] += 1
      else:
        geo_file.loc[path[i], f"{which_time}"] += 1

  geo_file['utils'] = geo_file[[f"{i}" for i in range(1, 49)]].sum(axis=1)
  
  geo_file.to_csv(save_path, index = None)
  
  return geo_file
  

def get_dynamic_speed(geo_path, traj_path, save_path, dataset):
  time_str = "%Y-%m-%dT%H:%M:%SZ"

  # dataset = "Porto_Taxi"

  traj = pd.read_csv(traj_path)
  geo_file = pd.read_csv(geo_path)

  st = []
  pt = []
  tt = []

  for index, row in tqdm(traj.iterrows(), total = len(traj)):
    time = row['time_list'].split(',')
    times = [datetime.strptime(_, time_str) for _ in time]
    st.append(times[0])
    tt.append((times[-1]-times[0]).total_seconds())
    pt.append([(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])])

  traj['st'] = st
  traj['pt'] = pt
  traj['tt'] = tt
    
  speed_dict = {
      outer_key: {inner_key: (0, 0) for inner_key in range(1, 49)} 
      for outer_key in range(len(geo_file))
  }

  for index, row in tqdm(traj.iterrows(), total=len(traj)):
    path = eval(row['rid_list'])
    time = row["time_list"].split(',')
    time = [datetime.strptime(_, "%Y-%m-%dT%H:%M:%SZ") for _ in time]
    pt = row['pt']
    for i in range(len(path)-1):
      if pt[i] > 0:
        speed = geo_file.loc[path[i], 'length'] / pt[i]
        if speed < 35:
          which_time = time[i].hour + 1
          avg, n = speed_dict[path[i]].get(which_time, (0, 0))
          speed_dict[path[i]][which_time] = ((avg * n + speed) / (n + 1), (n + 1))  
    
  with open(save_path, mode='w', newline='') as file:
      writer = csv.writer(file)
      
      writer.writerow(['geo_id'] + list(int(i) for i in speed_dict[1].keys())) 
      
      for outer_key, inner_dict in speed_dict.items():
          inner = list(i[0] for i in inner_dict.values())
          row = [outer_key] + inner
          writer.writerow(row)




    
