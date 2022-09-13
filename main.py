import os
import time
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import move_data_to_device
from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.io as scio
from pprint import pformat

from src import metric_fn
from src.utils import init_logger, logger
from src.dataloader import CVDataset, DRDataset
from src.model import CSL

#!/usr/bin/env python
# coding=utf-8

@torch.no_grad()    # @torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
def train_test_fn(model, train_loader, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in train_loader:
        model.train_step(batch)
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.test_step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp - eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                                 "col": edges[1],
                                 "score": scores,
                                 "label": labels,
                                 })
        logger.info(f"save time cost: {time.time() - eval_end_time_stamp}")
    return scores, labels, edges, metric

@torch.no_grad()
def test_fn(model, val_loader, save_file_format=None):
    device = model.device
    state = model.training
    model.eval()
    scores, labels, edges = [], [], []
    for batch in val_loader:
        batch = move_data_to_device(batch, device)
        output = model.step(batch)
        label, score = output["label"], output["predict"]
        edge = batch.interaction_pair[:, batch.valid_mask.reshape(-1)]
        scores.append(score.detach().cpu())
        labels.append(label.cpu())
        edges.append(edge.cpu())
    model.train(state)
    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    edges = torch.cat(edges, dim=1).numpy()
    eval_star_time_stamp = time.time()
    metric = metric_fn.evaluate(predict=scores, label=labels)
    eval_end_time_stamp = time.time()
    logger.info(f"eval time cost: {eval_end_time_stamp-eval_star_time_stamp}")
    if save_file_format is not None:
        save_file = save_file_format.format(aupr=metric["aupr"], auroc=metric["auroc"])
        scio.savemat(save_file, {"row": edges[0],
                      "col": edges[1],
                      "score": scores,
                      "label": labels,
                      })
        logger.info(f"save time cost: {time.time()-eval_end_time_stamp}")
    return scores, labels, edges, metric


def train_fn(config, model, train_loader, val_loader):

    # pl.callbacks.LearningRateMonitor("epoch"): set to 'epoch' or 'step' to log lr of all optimizers at the same interval,
    # pl.callbacks.LearningRateMonitor("epoch"): set to None to log at individual interval according to the interval key of each scheduler. Defaults to None.
    lr_callback = pl.callbacks.LearningRateMonitor("epoch")

    trainer = Trainer(max_epochs=config.epochs,         # Force training for at least these many epochs. Disabled by default (None).
                      default_root_dir=config.log_dir,  # Default path for logs and weights when no logger/ckpt_callback passed.
                      profiler=config.profiler,         # To profile individual steps during training and assist in identifying bottlenecks. Default: None.
                      fast_dev_run=False,               # Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test). Default: False.
                      callbacks=[lr_callback],          # Add a callback or list of callbacks. Default: None.
                      gpus=config.gpus,                 # Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node Default: None.
                      check_val_every_n_epoch=1)        # Check val every n train epochs. Default: 1.

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)   # Runs the full optimization routine.

    if not hasattr(config, "dirpath"):
        config.dirpath = trainer.checkpoint_callback.dirpath
    print(model.device)


def train(config, model_cls=CSL):
    time_stamp = time.asctime()     # 返回一个可读的形式为"Tue Dec 11 18:07:14 2008"的24个字符的字符串。
    datasets = DRDataset(dataset_name=config.dataset_name,
                         drug_neighbor_num=config.drug_neighbor_num,
                         disease_neighbor_num=config.disease_neighbor_num)

    # log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}", f"{model_cls.__name__}", f"{time_stamp}")
    log_dir = os.path.join(f"{config.comment}", f"{config.split_mode}-{config.n_splits}-fold", f"{config.dataset_name}", f"{model_cls.__name__}")
    config.log_dir = log_dir

    config.n_drug = datasets.drug_num
    config.n_disease = datasets.disease_num

    config.size_u = datasets.drug_num
    config.size_v = datasets.disease_num
    config.pos_weight = datasets.pos_weight

    config.gpus = 1 if torch.cuda.is_available() else 0
    config.time_stamp = time_stamp

    logger = init_logger(log_dir)
    logger.info(pformat(vars(config)))

    config.dataset_type = config.dataset_dype if config.dataset_type is not None else model_cls.DATASET_TYPE
    cv_spliter = CVDataset(datasets,
                           split_mode=config.split_mode,
                           n_splits=config.n_splits,
                           drug_idx=config.drug_idx,
                           disease_idx=config.disease_idx,
                           global_test_all_zero=config.global_test_all_zero,
                           train_fill_unknown=config.train_fill_unknown,
                           dataset_type=config.dataset_type,
                           seed=config.seed)

    pl.seed_everything(config.seed)
    scores, labels, edges, split_idxs = [], [], [], []
    metrics = {}
    start_time_stamp = time.time()


    for split_id, datamodule in enumerate(cv_spliter):
        # if split_id not in [4, 5]:
        #     continue
        config.split_id = split_id
        split_start_time_stamp = time.time()

        datamodule.prepare_data()
        train_loader = datamodule.train_dataloader()    # 准备训练数据
        val_loader = datamodule.val_dataloader()        # 准备测试数据
        config.pos_weight = train_loader.dataset.pos_weight

        model = model_cls(**vars(config))               # 初始化模型
        model = model.cuda() if config.gpus else model

        if split_id==0:
            logger.info(model)

        logger.info(f"begin train fold {split_id}/{len(cv_spliter)}")
        train_fn(config, model, train_loader=train_loader, val_loader=val_loader)
        logger.info(f"end train fold {split_id}/{len(cv_spliter)}")
        save_file_format = os.path.join(config.log_dir,f"{config.dataset_name}-{config.split_id} fold-{{auroc}}-{{aupr}}.mat")
        score, label, edge, metric = test_fn(model, val_loader, save_file_format)
        # score, label, edge, metric = train_test_fn(model, train_loader, val_loader, save_file_format)

        metrics[f"split_id_{split_id}"] = metric
        scores.append(score)
        labels.append(label)
        edges.append(edge)
        split_idxs.append(np.ones(len(score), dtype=int)*split_id)

        logger.info(f"{split_id}/{len(cv_spliter)} folds: {metric}")
        logger.info(f"{split_id}/{len(cv_spliter)} folds time cost: {time.time()-split_start_time_stamp}")

        if config.debug:
            break


    end_time_stamp = time.time()
    logger.info(f"total time cost:{end_time_stamp-start_time_stamp}")
    with pd.ExcelWriter(os.path.join(log_dir, f"tmp.xlsx")) as f:
        pd.DataFrame(metrics).T.to_excel(f, sheet_name="metrics")
        params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
        params.to_excel(f, sheet_name="params")

    scores = np.concatenate(scores, axis=0)
    labels = np.concatenate(labels, axis=0)
    edges = np.concatenate(edges, axis=1)
    split_idxs = np.concatenate(split_idxs, axis=0)
    final_metric = metric_fn.evaluate(predict=scores, label=labels, is_final=True)
    metrics["final"] = final_metric
    metrics = pd.DataFrame(metrics).T
    metrics.index.name = "split_id"
    metrics["seed"] = config.seed
    logger.info(f"final {config.dataset_name}-{config.split_mode}-{config.n_splits}-fold-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}")


    # # 将结果输出为表格文件
    # output_file_name = f"final-{config.dataset_name}-{config.split_mode}-{config.n_splits}-auroc:{final_metric['auroc']}-aupr:{final_metric['aupr']}-fold"
    # scio.savemat(os.path.join(log_dir, f"{output_file_name}.mat"), {"row": edges[0], "col": edges[1], "score": scores, "label": labels, "split_idx":split_idxs})
    # with pd.ExcelWriter(os.path.join(log_dir, f"{output_file_name}.xlsx")) as f:
    #     metrics.to_excel(f, sheet_name="metrics")
    #     params = pd.DataFrame({key:str(value) for key, value in vars(config).items()}, index=[str(time.time())])
    #     for key, value in final_metric.items():
    #         params[key] = value
    #     params["file"] = output_file_name
    #     params.to_excel(f, sheet_name="params")
    # logger.info(f"save final results to r'{os.path.join(log_dir, output_file_name)}.mat'")


    logger.info(f"final results: {final_metric}")



def parse(print_help=False):
    # 1、创建一个解析器——创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser()

    # 2、添加参数——给一个 ArgumentParser 添加程序参数信息
    parser.add_argument("--model", default="CSL", type=str)
    parser.add_argument("--epochs", default=1, type=int, help = 'Default is 64 or 49.')
    parser.add_argument("--drug_feature_topk", default=20, type=int)    # add prior knowledge
    parser.add_argument("--disease_feature_topk", default=20, type=int)
    parser.add_argument("--debug", default=True, action="store_true")   # 当设置为False时将进行n折交叉验证，当设置为True时仅计算1折就停止训练
    parser.add_argument("--profiler", default=False, type=str)
    parser.add_argument("--comment", default="result", type=str, help="The folder directory where the experimental log is saved.")
    parser = DRDataset.add_argparse_args(parser)
    parser = CVDataset.add_argparse_args(parser)
    parser = CSL.add_model_specific_args(parser)

    # 3、解析参数——使用 parse_args() 解析添加的参数
    args = parser.parse_args()

    if print_help:
        parser.print_help()
    return args


if __name__=="__main__":
    args = parse(print_help=True)
    train(args)
