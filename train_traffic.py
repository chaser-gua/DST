import argparse
import logging
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset


def get_logger(log_dir: str, ver: str) -> logging.Logger:
    logger = logging.getLogger(f"run_{ver}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{current_time}_{ver}.log")

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FixedLengthDataset(Dataset):

    def __init__(self, data: torch.Tensor, seq_id: torch.Tensor, seq_cls: torch.Tensor, device: str):
        self.data = data
        self.seq_id = seq_id
        self.seq_cls = seq_cls
        self.device = device

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: int):
        src = self.data[idx]  # (24,)

        tgt = torch.cat([src[1:], torch.zeros(1, device=self.device)], dim=0)  # (24,)

        padding_mask = torch.zeros(24, dtype=torch.bool, device=self.device)
        padding_mask[-1] = True

        return src, tgt, padding_mask, self.seq_id[idx], self.seq_cls[idx]


class PositionalEncodingBatchFirst(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 30000):
        super(PositionalEncodingBatchFirst, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MaskedTransformer_1(nn.Module):
    def __init__(
        self,
        device: str,
        input_dim: int = 1,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        seq_len: int = 24,
        num_classes: int = 2,
    ):
        super(MaskedTransformer_1, self).__init__()
        self.device = device
        self.d_model = d_model
        self.seq_len = seq_len

        self.input_embed = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncodingBatchFirst(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.reg = nn.Linear(d_model, 1)
        self.cls = nn.Linear(d_model, num_classes)

        self.register_buffer("causal_mask", self._generate_causal_mask(seq_len))

    def _generate_causal_mask(self, sz: int) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):

        sequence_rep = self.encoder(
            x,
            mask=self.causal_mask,
            src_key_padding_mask=padding_mask,
        )  # (B, T, d_model)

        reg_predictions = self.reg(sequence_rep)  # (B, T, 1)

        # last position is masked/padded by design; use T-2 token
        token_rep = sequence_rep[:, -2, :]  # (B, d_model)
        cls_predictions = self.cls(token_rep)  # (B, num_classes)

        return reg_predictions, cls_predictions, token_rep


def masked_mse_1(pred: torch.Tensor, target: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    if target.dim() == 2:
        target = target.unsqueeze(-1)

    valid_mask = (~padding_mask).unsqueeze(-1)  # (B, T, 1)
    mse = (pred - target) ** 2
    masked_mse = mse * valid_mask.float()
    return masked_mse.sum() / (valid_mask.sum() + 1e-8)


def train(
    model: nn.Module,
    dataloader: DataLoader,
    logger: logging.Logger,
    dataset_name: str,
    ver: str,
    seed: int,
    lambda1: float = 10.0,
    device: str = "cuda:0",
    epochs: int = 100,
    lr: float = 1e-4,
) -> int:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    cls_criterion = nn.CrossEntropyLoss()

    best_epoch = 0
    best_loss = float("inf")

    save_dir = "./saved_model/{}/dy/seed{}/{}/".format(dataset_name, seed, ver)
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_loss1 = 0.0
        total_loss2 = 0.0

        for batch in dataloader:
            src, tgt, padding_mask, _, seq_cls = batch
            src = src.to(device)
            tgt = tgt.to(device)
            padding_mask = padding_mask.to(device)
            seq_cls = seq_cls.to(device)

            reg_predictions, cls_predictions, _ = model(src, padding_mask=padding_mask)

            loss_1 = masked_mse_1(reg_predictions, tgt, padding_mask)
            loss_2 = cls_criterion(cls_predictions, seq_cls)
            loss = loss_1 * lambda1 + loss_2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss1 += float(loss_1.item())
            total_loss2 += float(loss_2.item())
            total_loss += float(loss.item())

        avg_loss1 = total_loss1 / max(1, len(dataloader))
        avg_loss2 = total_loss2 / max(1, len(dataloader))
        avg_loss = total_loss / max(1, len(dataloader))

        logger.info(
            "Epoch {}, loss1: {:.6f}, loss2: {:.6f}, loss: {:.6f}".format(epoch, avg_loss1, avg_loss2, avg_loss)
        )

        if avg_loss < best_loss:
            best_epoch = epoch
            best_loss = avg_loss
            save_model_path = os.path.join(save_dir, "{}_epoch{}.pth".format(ver, best_epoch))
            torch.save(model, save_model_path)

    best_name = "{}_epoch{}.pth".format(ver, best_epoch)
    for filename in os.listdir(save_dir):
        if filename != best_name:
            try:
                os.remove(os.path.join(save_dir, filename))
            except OSError:
                pass

    return best_epoch


def main():
    seq_len = 24
    input_dim = 1
    max_epoch = 100

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Porto_Taxi")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ver", type=str, required=True)
    parser.add_argument("--isAug", action="store_true")
    parser.add_argument("--magicD", type=int, default=64)
    parser.add_argument("--lambda1", type=float, default=10.0)
    parser.add_argument("--ckpt", type=str, default="")

    args = parser.parse_args()

    dataset_name = args.dataset
    seed = args.seed
    device = args.device
    ver = args.ver
    is_aug = args.isAug
    batch_size = args.magicD
    lambda1 = args.lambda1

    setup_seed(seed)

    log_path = "./log/{}/reg/{}/".format(dataset_name, ver)
    logger = get_logger(log_dir=log_path, ver=ver)
    logger.info(
        "dataset={}, ver={}, batch_size={}, seed={}, device={}, isAug={}".format(
            dataset_name, ver, batch_size, seed, device, is_aug
        )
    )

    model = MaskedTransformer_1(device=device, input_dim=input_dim, seq_len=seq_len).to(device)

    if not is_aug:
        df = pd.read_csv("./{}/time_utils.csv".format(dataset_name))
        dv_wd = df.iloc[:, 2:26].astype(float)
        dv_we = df.iloc[:, 26:-1].astype(float)
    else:
        df = pd.read_csv("./{}/dynamics.csv".format(dataset_name))
        dv_wd = df.iloc[:, 2:26].astype(float)
        dv_we = df.iloc[:, 26:-1].astype(float)

    scaler = MinMaxScaler()
    dv_wd = torch.FloatTensor(scaler.fit_transform(dv_wd.T).T).to(device)  # (N, 24)
    dv_we = torch.FloatTensor(scaler.fit_transform(dv_we.T).T).to(device)  # (N, 24)

    data_x = torch.cat([dv_wd, dv_we], dim=0)  # (2N, 24)

    n = dv_wd.shape[0]
    seq_id = torch.arange(n, dtype=torch.long, device=device).repeat(2)  # (2N,)
    seq_cls = torch.LongTensor([i for i in range(2) for _ in range(n)]).to(device)  # (2N,)

    datasets = FixedLengthDataset(data_x, seq_id, seq_cls, device=device)
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

    if args.ckpt:
        logger.info("Loading checkpoint: {}".format(args.ckpt))
        model = torch.load(args.ckpt, map_location=device)
        model.eval()
    else:
        best_epoch = train(
            model=model,
            dataloader=dataloader,
            logger=logger,
            dataset_name=dataset_name,
            ver=ver,
            seed=seed,
            lambda1=lambda1,
            device=device,
            epochs=max_epoch,
        )
        logger.info("best_epoch: {}".format(best_epoch))
        best_path = "./saved_model/{}/dy/seed{}/{}/{}_epoch{}.pth".format(dataset_name, seed, ver, ver, best_epoch)
        model = torch.load(best_path, map_location=device)
        model.eval()

    dy_rep = defaultdict(list)
    dy_reps = []

    with torch.no_grad():
        for batch in dataloader:
            src, _, padding_mask, batch_seq_id, _ = batch
            src = src.to(device)
            padding_mask = padding_mask.to(device)

            _, _, rep = model(src, padding_mask=padding_mask)  # (B, d_model)

            for i, node_id in enumerate(batch_seq_id):
                dy_rep[int(node_id)].append(rep[i].unsqueeze(0))

    for _, v in dy_rep.items():
        dy_reps.append(torch.cat(v, dim=0).mean(dim=0, keepdim=True))

    dy_reps = torch.cat(dy_reps, dim=0).to(device)  # (N, d_model)
    logger.info("dy_reps shape: {}".format(tuple(dy_reps.shape)))

    os.makedirs("./saved_model/{}/emb/seed{}/".format(dataset_name, seed), exist_ok=True)
    out_path = "./saved_model/{}/emb/seed{}/{}.npy".format(dataset_name, seed, ver)
    np.save(out_path, dy_reps.detach().cpu().numpy())
    logger.info("Saved embeddings: {}".format(out_path))


if __name__ == "__main__":
    main()
