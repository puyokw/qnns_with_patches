#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import socket

import math
import numpy as np
import pandas as pd
#import seaborn as sns

#from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
#from torch.optim import Adam, SGD
import torch.utils.checkpoint as checkpoint
#from torch.distributed.optim import DistributedOptimizer
#import torch.distributed.rpc as rpc
#from torch.distributed.rpc import RRef

import torchvision
from torchvision.transforms import v2

import warnings
import time
#import functools
#import copy
#from tqdm import tqdm

# from pyhessian import hessian

#import sys
#import tempfile
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI

seed = 1001
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)
warnings.simplefilter('ignore', UserWarning)

import torchquantum as tq
from torchquantum.measurement import expval_joint_analytical


n_qubits = 8
n_depth_per_block = 200
n_qnn = 16
max_epochs = 50
coeff=100

n_half_qubits = n_qubits//2 
n_latter_half_qubits = n_qubits-n_half_qubits

data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
data_tr_cpu, label_tr_cpu = data.train_data, data.train_labels
data_tr_cpu = data_tr_cpu/255*2*math.pi/n_qubits
data_tr_cpu = torch.nn.AvgPool2d( (2,2), stride=(2,2) )(data_tr_cpu) # (28,28) -> (14,14)

dataset_name = 'mnist'
n_class = len(np.unique(label_tr_cpu))
#print(data_tr_cpu.shape, label_tr_cpu.shape)
#print(f"n_class: {n_class}")

### pyhessian のコードをddp に拡張 ###
def get_params_grad_ddp(model_ddp):
    params = []
    grads = []
    # DDPモデルの場合は .module で元のモデルにアクセス
    actual_model = model_ddp.module if isinstance(model_ddp, DDP) else model_ddp
    for param in actual_model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        # DDPでは勾配は各プロセスで計算され、all-reduceされるが、
        # backward()直後にアクセスすればローカルな勾配があるはず
        # ただし、DDPの内部実装によってはNoneの場合もあるかもしれない
        grads.append(0. if param.grad is None else param.grad.clone() + 0.)
    return params, grads

# ベクトルのリストの内積を計算し、全プロセスで合計 (AllReduce)
def group_product_ddp(xs, ys):
    local_product = sum([torch.sum(x * y) for (x, y) in zip(xs, ys) if x is not None and y is not None])
    total_product = local_product.clone()
    dist.all_reduce(total_product, op=dist.ReduceOp.SUM)
    return total_product

# ベクトルのリストを正規化 (ランク0でノルム計算 -> ブロードキャスト -> 各プロセスで正規化)
def normalization_ddp(v, rank, device):
    # ノルムの二乗を計算して集約
    s_sq = group_product_ddp(v, v) # includes all_reduce operation
    # ランク0でのみ平方根を計算 (各々で計算してもよいかも)
    s = torch.tensor(0.0, device=device)
    if rank == 0:
        s = torch.sqrt(s_sq)
    # ランク0から全プロセスにノルム s をブロードキャスト
    dist.broadcast(s, src=0)
    # 各プロセスで正規化
    s_item = s.item() + 1e-6 # Avoid division by zero
    v_normalized = [(vi / s_item).clone() for vi in v] # clone() で新しいテンソルを作成
    return v_normalized

# ベクトルのリスト w を v_list に対して直交化し、正規化 (DDP対応)
def orthnormal_ddp(w, v_list, rank, device):
    for v in v_list:
        # 内積を計算して集約
        prod = group_product_ddp(w, v) # includes all_reduce operation
        # 各プロセスで w = w - prod * v を計算
        for i in range(len(w)):
            if w[i] is not None and v[i] is not None: # None をゼロにするかどうか
                w[i].data.add_(v[i], alpha=-prod) # インプレース操作
    return normalization_ddp(w, rank, device) # includes broadcast operation

class HessianDDP():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model_ddp, criterion, dataloader, device, rank, world_size):
   
        self.model = model_ddp.eval()  # make model is in evaluation model
        self.criterion = criterion
        self.rank = rank
        self.world_size = world_size
        self.dataloader = dataloader
        self.device = device # cuda:0, 各node のcuda:0
        self.params, _ = get_params_grad_ddp(self.model)

    def dataloader_hv_product(self, v):
        """
        データローダー全体を使って Hessian-vector product (Hv) を計算 (DDP対応)
        v: 対象のベクトル (テンソルのリスト)
        """
        self.model.zero_grad() # 計算前にモデルの勾配をリセット
        num_data_local = torch.tensor(0.0, device=self.device)
        THv_local = [torch.zeros_like(p, device=self.device) for p in self.params]

        for inputs, targets in self.dataloader: 
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)

            # DDPモデルはフォワード時に内部で同期を行う
            # print(inputs.shape, inputs.dtype)
            self.model.zero_grad() # autograd.grad のために勾配をクリア
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            params_list, gradsH_list = get_params_grad_ddp(self.model) # パラメータと現在の勾配(ここでは使わない)
            loss.backward(create_graph=True) # Hv計算のためにグラフを保持
            params_list, gradsH_list = get_params_grad_ddp(self.model) # ここで勾配を取得
            self.model.zero_grad() # 次のバッチのためにクリア
            # gradsH (テンソルのリスト) と v (テンソルのリスト) を使って Hv を計算
            valid_gradsH = [g for g in gradsH_list if g is not None]
            valid_params = [p for p, g in zip(params_list, gradsH_list) if g is not None]
            valid_v = [v_i for v_i, g in zip(v, gradsH_list) if g is not None]
            if not valid_gradsH: # 勾配が全くない場合スキップ
                continue
            Hv_local_batch = torch.autograd.grad(
                outputs=valid_gradsH,
                inputs=valid_params,
                grad_outputs=valid_v,
                only_inputs=True,
                retain_graph=False # この autograd.grad ではグラフ不要
            )
            # ローカルな結果を蓄積
            for i, hv_b in enumerate(Hv_local_batch):
                # if hv_b is not None: # autograd.grad は None を返すことがある
                # 元のパラメータリストに対応する位置に加算
                # param_index = self.params.index(valid_params[i]) # ちょっと非効率かも
                THv_local[i] = THv_local[i]+hv_b * batch_size + 0.
            num_data_local += batch_size
        # --- ローカル計算結果を全プロセスで集約 (AllReduce) ---
        # 合計データ数を集約 
        dist.all_reduce(num_data_local, op=dist.ReduceOp.SUM)
        total_data = num_data_local.item()
        # THv (Hvの合計) を集約
        for thv in THv_local:
            dist.all_reduce(thv, op=dist.ReduceOp.SUM)
        # 平均化
        THv = [(thv / total_data).clone() for thv in THv_local] # cloneして新しいリストを作成
        # 固有値要素 (v^T * H * v) を計算して集約
        eigenvalue_local = group_product_ddp(THv, v) # この関数内でall_reduceされる
        return eigenvalue_local.item(), THv # 集約済みの固有値と平均化されたHvを返す

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1, start_iter=0):
        """
        Top N 固有値と固有ベクトルを計算 (DDP対応 Power Iteration)
        """
        assert top_n >= 1
        eigenvalues = []
        eigenvectors = [] # 計算された固有ベクトルを格納

        for k in range(top_n):
            eigenvalue = None
            # --- 1. 初期ベクトル v の生成と同期 ---
            v = None # 全プロセスで定義
            shapes = [p.shape for p in self.params] # 全プロセスでパラメータ形状は共有されていると仮定
            numels = [p.numel() for p in self.params]
            if self.rank == 0:
                v_rank0 = [torch.randn_like(p, device=self.device) for p in self.params]
                s_sq = sum([torch.sum(vi *vi) for vi in v_rank0])
                s = torch.sqrt(s_sq) + 1e-6
                s_item = s.item()
                v_list = [(vi / s_item).clone() for vi in v_rank0]
                # post processing 
                # 2. テンソルリストを単一のフラットなテンソルに連結
                v_flat = torch.cat([t.view(-1) for t in v_list])
                #print(f"rank {self.rank}: before broadcast flat vector (size: {v_flat.numel()})")
                dist.broadcast(v_flat, src=0)
                #print(f"rank {self.rank}: flat vector broadcasted")
                # 4. (ランク0でも) フラットベクトルからリストを復元 (必須ではないが一貫性のため)
                v_split = torch.split(v_flat, numels)
                v = [t.view(shape) for t, shape in zip(v_split, shapes)]
                if start_iter>0 and os.path.exists(f"init_v_rank0_{start_iter-1}.pt"):
                    path_name = f'init_v_rank0_list_{start_iter-1}th_iteration.pt'
                    v = torch.load(path_name, map_location=self.device)
            else:
                # 他のランクはランク0と同じ形状のゼロベクトルを準備
                v = [torch.zeros_like(p, device=self.device) for p in self.params]
                # 1. 他のランクは受け皿となるフラットベクトルを準備
                #    形状(shapes)と要素数(numels)はランク0と同じはず
                total_numel = sum(numels)
                #    データ型をランク0のパラメータに合わせる (ここでは params[0] から取得)
                v_flat = torch.zeros(total_numel, dtype=self.params[0].dtype, device=self.device)
                # 2. フラット化されたベクトルを受信
                #print(f"rank {self.rank}: before broadcast flat vector (expecting size: {total_numel})")
                dist.broadcast(v_flat, src=0) # v_flat が受信データで上書きされる
                #print(f"rank {self.rank}: flat vector received")
                # 3. フラット化されたベクトルからリストを復元
                #    torch.splitで各テンソルの要素数に基づいて分割
                v_split = torch.split(v_flat, numels)
                #    view()で元の形状に戻す
                v = [t.view(shape) for t, shape in zip(v_split, shapes)]

            # --- 2. Power Iteration ---
            for i in range(start_iter, maxIter):
                path_name = f'init_v_rank0_list_{i}th_iteration.pt'
                if self.rank == 0:
                    print(f"rank {self.rank}: {i}th iteration")
                start_time = MPI.Wtime()
                # 計算済みの固有ベクトルに対して直交化 (v_list = eigenvectors)
                if eigenvectors: # 最初の固有ベクトル計算時は不要
                    v = orthnormal_ddp(v, eigenvectors, self.rank, self.device)
                    # 直交化後の v を同期 (orthnormal_ddp内で正規化とbroadcastが行われる)

                # --- 3. Hv の計算 (DDP対応) ---
                # dataloader_hv_product は内部で all_reduce を行う
                # 戻り値 tmp_eigenvalue は集約済み、Hv は平均化済み
                # print("3 calculating data_loader_hv_product")
                tmp_eigenvalue, Hv = self.dataloader_hv_product(v)

                # --- 4. v の更新と同期 ---
                # Hv を正規化して新しい v とする (ランク0で計算しブロードキャスト)
                v = normalization_ddp(Hv, self.rank, self.device) # 結果は全プロセスで同期される

                # --- 5. 収束判定 (ランク0で行い結果をブロードキャスト) ---
                converged = torch.tensor(0, device=self.device, dtype=torch.int)
                if self.rank == 0:
                    print(f"  Iteration {i}: eigenvalue = {tmp_eigenvalue:.6e}") # 進捗表示
                    if eigenvalue is not None:
                        if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                            converged = torch.tensor(1, device=self.device, dtype=torch.int)
                    eigenvalue = tmp_eigenvalue # 更新

                # 収束判定結果をブロードキャスト
                dist.broadcast(converged, src=0)
                # debugging info
                end_time = MPI.Wtime()
                duration = end_time - start_time
                if self.rank == 0:
                    print(f"{i}th iteration of PowerIteration: Rank {self.rank}: Duration: {duration:.6f}s")
                    max_allocated = torch.cuda.max_memory_allocated(self.device)
                    free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                    print(f"Max Allocated: {max_allocated / 1024**2:.2f} MiB")
                    print(f"Free:  {free_mem / 1024**2:.2f} MiB")  # デバイス全体の空き容量
                    print(f"Total: {total_mem / 1024**2:.2f} MiB") # デバイス全体の総容量
                    torch.save(v_rank0, path_name)
                if converged.item() == 1:
                    if self.rank == 0:
                         print(f"Converged at iteration {i}")
                    break # 全プロセスでループを抜ける
            # --- Power Iteration 終了 ---

            # 最終的な固有値と固有ベクトルを格納 (ランク0のみ)
            if self.rank == 0:
                 print(f"Top {k+1} eigenvalue: {eigenvalue:.6e}")
                 eigenvalues.append(eigenvalue)
            # 固有ベクトル v は全プロセスで同期されているので、そのままリストに追加
            eigenvectors.append(v)

        # ランク0のみ結果を返す (他のランクは空リスト)
        if self.rank == 0:
            return eigenvalues, eigenvectors
        else:
            return [], []

###
class ConstCoeffLayer(nn.Module):
    def __init__(self, coeff):
        super().__init__()
        self.coeff = coeff
    def forward(self, x):
        ret = x * self.coeff
        return ret

def calc_exp_val(qdev, obs):
    assert len(obs)==n_qubits
    state2 = qdev.states.clone()
    for i in range(n_qubits):
        if obs[i]=='I':
            continue
        elif obs[i]=='X':
            mat = torch.tensor([[0,1],[1,0]])
        elif obs[i]=='Y':
            mat = torch.tensor([[0,-1j],[1j,0]])
        elif obs[i]=='Z':
            mat = torch.tensor([[1,0],[0,-1]])
        state2 = tq.functional.apply_unitary_bmm(state2, mat, [i])
    state1 = qdev.states.clone()
    exp_val = torch.einsum("bij...k,bij...k->b", state1.conj(), state2).real
    return exp_val

# 14x14 => 7x14x2
# 2n_qubitsx28 => 14x7x8 = 14x28x2
class QNNsubModel(nn.Module):
    def __init__(self):
        # params is numpy array
        super().__init__()

    def forward(self, x, phi):
        bsz, nx_features = x.shape
        qdev = tq.QuantumDevice(
            n_wires=n_qubits, bsz = bsz, device=x.device, record_op=False
        )
        for k in range(n_depth_per_block):
            # j = 2*d*n_depth_per_block + 2*k
            for i in range(n_qubits):
                tq.functional.rx(qdev, wires=i, params=phi[i+2*k*n_qubits])
            for i in range(n_qubits):
                tq.functional.ry(qdev, wires=i, params=phi[i+(2*k+1)*n_qubits])
            for i in range(n_qubits):
                qdev.cz(wires=[i,(i+1)%n_qubits])
        for i in range(n_qubits): # x: 32, phi: 64
            for j in range(n_half_qubits):
                if j%2==0:
                    tq.functional.ry(qdev, wires=i, params=phi[2*n_half_qubits*i+2*j+2*n_depth_per_block*n_qubits])
                    tq.functional.rx(qdev, wires=i, params=x[:,n_half_qubits*i+j]) ##
                    tq.functional.ry(qdev, wires=i, params=phi[2*n_half_qubits*i+2*j+1+2*n_depth_per_block*n_qubits])
                else:
                    tq.functional.rx(qdev, wires=i, params=phi[2*n_half_qubits*i+2*j+2*n_depth_per_block*n_qubits])
                    tq.functional.ry(qdev, wires=i, params=x[:,n_half_qubits*i+j]) ##
                    tq.functional.rx(qdev, wires=i, params=phi[2*n_half_qubits*i+2*j+1+2*n_depth_per_block*n_qubits])
        for i in range(n_qubits):
            qdev.cz(wires=[i,(i+1)%(n_qubits)])
        for k in range(n_depth_per_block):
            # j = 2*d*n_depth_per_block + 2*k
            for i in range(n_qubits):
                tq.functional.rx(qdev, wires=i, params=phi[i+(2*n_depth_per_block+2*n_half_qubits +2*k)*n_qubits])
            for i in range(n_qubits):
                tq.functional.ry(qdev, wires=i, params=phi[i+(2*n_depth_per_block+2*n_half_qubits +2*k+1)*n_qubits])
            for i in range(n_qubits):
                qdev.cz(wires=[i,(i+1)%n_qubits])
        for i in range(n_qubits): # 32, 64
            for j in range(n_latter_half_qubits):
                if j%2==0:
                    tq.functional.ry(qdev, wires=i, params=phi[2*n_latter_half_qubits*i+2*j+2*n_half_qubits*n_qubits+4*n_depth_per_block*n_qubits])
                    tq.functional.rx(qdev, wires=i, params=x[:,n_latter_half_qubits*i+j+n_half_qubits*n_qubits]) ##
                    tq.functional.ry(qdev, wires=i, params=phi[2*n_latter_half_qubits*i+2*j+1+2*n_half_qubits*n_qubits+4*n_depth_per_block*n_qubits])
                else:
                    tq.functional.rx(qdev, wires=i, params=phi[2*n_latter_half_qubits*i+2*j+2*n_half_qubits*n_qubits+4*n_depth_per_block*n_qubits])
                    tq.functional.ry(qdev, wires=i, params=x[:,n_latter_half_qubits*i+j+n_half_qubits*n_qubits]) ##
                    tq.functional.rx(qdev, wires=i, params=phi[2*n_latter_half_qubits*i+2*j+1+2*n_half_qubits*n_qubits+4*n_depth_per_block*n_qubits])
        for i in range(n_qubits):
            qdev.cz(wires=[i,(i+1)%(n_qubits)])
        j= 2
        for k in range(n_depth_per_block):
            for i in range(n_qubits):
                tq.functional.rx(qdev, wires=i, params=phi[i+(4*n_depth_per_block+2*n_qubits +2*k)*n_qubits])
            for i in range(n_qubits):
                tq.functional.ry(qdev, wires=i, params=phi[i+(4*n_depth_per_block+2*n_qubits +2*k+1)*n_qubits])
            if (k==n_depth_per_block-1):
                break
            for i in range(n_qubits):
                qdev.cz(wires=[i,(i+1)%n_qubits])
        global n_class
        obs_list = [ calc_exp_val(qdev, "I"*i+Pauli+"I"*(n_qubits-1-i)) for Pauli in ["X","Z"] for i in range(n_class//2)]
        ret = torch.stack(obs_list, dim=1)
        return ret


class QnnMpModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_gpus = dist.get_world_size()
        self.rank = dist.get_rank() # world rank
        self.device = torch.device(f"cuda:0")
        self.params_list = nn.ParameterList([torch.rand( (3*2*n_depth_per_block+2*n_qubits)*n_qubits, device='cuda:0' )*math.pi for _ in range(n_qnn)])
        # self.params_list = nn.ParameterList([torch.rand( (3*2*n_depth_per_block+2*n_qubits)*n_qubits )*math.pi for _ in range(n_qnn)])
        self.pos_bias = nn.Parameter(torch.zeros(14, 14))

    def forward(self, x):
        current_device = x.device
        # メモリ消費は激しいが実装の容易なdata 分割のみから
        n_data = len(x)
        in_x = x.to(current_device) + self.pos_bias.to(current_device) # local rank
        in_x = torch.stack([ in_x[:,i:i+n_qubits,j:j+n_qubits].reshape(n_data,n_qubits*n_qubits) for i in [0,2,4,6] for j in [0,2,4,6] ], axis=0) # (16,n_data,64)
        ret_list = [checkpoint.checkpoint(QNNsubModel(), in_x[i], self.params_list[i], use_reentrant=False) for i in range(n_qnn)]
        # ret_list = [QNNsubModel()(in_x[i], self.params_list[i]) for i in range(n_qnn)]
        ret = torch.stack(ret_list, axis=1) # (bsz, n_qnn, n_class)
        ret = torch.mean(ret, axis=1) # (bsz,n_class)
        return ret # on x.device 


def load_data(world_rank, world_size, data_tr_cpu, label_tr_cpu):
    device = torch.device("cuda:0") # 各nodeに1GPUなので、local rank. 
    dataloader = []
    n_split = 500 # 60000/500=120 # 1000
    tmp_skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=0)
    for i, (_, tmp_te) in enumerate(tmp_skf.split(data_tr_cpu, label_tr_cpu)):
        if i%world_size==world_rank:
            dataloader.append( (data_tr_cpu[tmp_te].to(device=device), label_tr_cpu[tmp_te].to(device=device)) )
    return dataloader

def setup(world_rank, world_size):
    comm = MPI.COMM_WORLD
    addr_data = None
    master_addr = None
    master_port = "29500"
    if world_rank==0:
        hostname = socket.gethostname()
        master_addr = socket.gethostbyname(hostname)
        print(f"Rank 0: Hostname={hostname}, Master Addr={master_addr}")
        addr_data = {'addr': master_addr}
    addr_data = comm.bcast(addr_data, root=0)
    master_addr = addr_data['addr']
    dist_url = f'tcp://{master_addr}:{master_port}'
    if world_rank==0:
        print(f"Rank {world_rank}: Received Master Addr={master_addr}")
        print(dist_url)
    dist.init_process_group(backend='nccl', init_method=dist_url,rank=world_rank, world_size=world_size)


def run(world_rank, world_size, data_tr_cpu, label_tr_cpu):
    global n_class
    comm = MPI.COMM_WORLD
    local_rank = 0
    # print(f"world rank: {world_rank}, local rank: {local_rank}")
    torch.cuda.set_device(local_rank) # このプロセスが使用するGPUをrank番に設定
    # print(f"world rank: {world_rank}, local rank: {local_rank}, Default device: {torch.cuda.current_device()}")
    setup(world_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")
    map_location = {'cuda:0': f'cuda:{local_rank}'}
    dataloader = load_data(world_rank, world_size, data_tr_cpu, label_tr_cpu)

    model = torch.nn.Sequential( QnnMpModel(), ConstCoeffLayer(coeff)).to(device)
    # --- load checkpoint ---
    dir_name = f"tmp_{n_qubits}qubits_{n_qnn}qnn"
    if n_depth_per_block > 50:
        dir_name += str(n_depth_per_block)
    prefix_name = f"{dataset_name}_{n_qnn}qnn{n_depth_per_block}_c{coeff}_{n_qubits}qubits_ensembling_cos"

    # 損失ログから最小損失のエポックを見つける (ランク0のみで実行)
    min_epochs = 0
    if world_rank == 0:
        losses_path = dir_name+'/'+prefix_name+'_losses.csv'
        losses = pd.read_csv(losses_path)
        min_index = losses['train_loss'].argmin()
        min_epochs = losses['epochs'][min_index]
        min_epochs = min_epochs.item()
        print(f"Loading checkpoint from epoch: {min_epochs} (min train loss)")
    # min_epochs を全プロセスにブロードキャスト
    #min_epochs_tensor = torch.tensor(min_epochs, dtype=torch.int, device=device)
    #dist.broadcast(min_epochs_tensor, src=0)
    # min_epochs = min_epochs_tensor.item()
    min_epochs = comm.bcast(min_epochs, root=0)
    model.load_state_dict(torch.load(dir_name+'/'+prefix_name+'_epoch'+str(min_epochs)+'.pt', map_location=map_location, weights_only=True))
    # ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    # print(f"world rank: {world_rank}, min_epochs: {min_epochs}")
    if world_rank == 0:
        print("calculating Hessian")
    model.eval()
    stime = time.perf_counter()
    # DDP対応のHessianクラスを使用
    hessian_comp = HessianDDP(
        model_ddp=model,
        criterion=torch.nn.CrossEntropyLoss(), # 損失関数
        dataloader=dataloader,                 # DDP対応データローダー
        device=device,                         # 現在のデバイス
        rank=world_rank,                             # 現在のランク
        world_size=world_size                  # 全プロセス数
    )
    # print(f"rank {world_rank}, hessain class defined")
    top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n=1)
    # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(maxIter=1, top_n=1) # for debugging 
    etime = time.perf_counter()
    dist.destroy_process_group() # cleanup()
    if world_rank==0:
        print('elapsed time: ', etime-stime, 'sec')
        print('Top Eigenvalues: ', top_eigenvalues)
        pd.DataFrame({"top eigenvalues": top_eigenvalues}).to_csv(prefix_name+"_top_eigen_values.csv", index=False)



if __name__ == '__main__':
    world_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # world_size = comm.Get_size()
    os.environ['RANK']=os.environ['OMPI_COMM_WORLD_RANK']
    if world_rank==0:
        print(torch.__version__)
        print(tq.__version__)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    processes = []
    run(world_rank, world_size, data_tr_cpu, label_tr_cpu)

