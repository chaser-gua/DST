# <div align="center"> Dual-branch Spatial-Temporal Self-supervised Representation for Enhanced Road Network Learning </div>

The official implementation of the "Dual-branch Spatial-Temporal self-supervised representation framework", which is accepted by AAAI 2026 (AI for Social Impact Track)

## Overview of the framework

![image](https://github.com/chaser-gua/DST/blob/master/framework.png)

The overview of the proposed DST framework. The high-order relationships are modeled via mix-hop transition matrix weighting and multi-view graph contrastive learning. The temporal travel traffic dynamics are integrated by the Transformer with two specific task-driven updates. Both block co-enhanced representations power downstream tasks jointly.

* Spatial Semantic Graphs Training

```
python train_graph.py
```

--dataset specifies the dataset, such as Beijing, Porto, or San Francisco

--seed specifies the random seed

--cuda specifies the GPU device number

--ver specifies the version

* Temporal Dynamics Training

```
python train_traffic.py
```

--dataset specifies the dataset, such as Beijing, Porto, or San Francisco

--seed specifies the random seed

--cuda specifies the GPU device number




  
