import os
import sys
import logging
import torch

logger = logging.getLogger("WBNCF")

class NoParsingFilter(logging.Filter):
    def filter(self, record):
        if record.funcName=="summarize" and record.levelno==20:
            return False
        if record.funcName=="_info" and record.funcName=="distributed.py" and record.lineno==20:
            return False
        return True

def init_logger(log_dir):
    lightning_logger = logging.getLogger("pytorch_lightning.core.lightning")
    lightning_logger.addFilter(NoParsingFilter())
    distributed_logger = logging.getLogger("pytorch_lightning.utilities.distributed")
    distributed_logger.addFilter(NoParsingFilter())
    format = '%Y-%m-%d %H-%M-%S'
    fm = logging.Formatter(fmt="[%(asctime)s][%(levelname)s] %(name)s: %(lineno)4d: %(message)s",
                           datefmt="%m/%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fm)
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    if len(logger.handlers)==1:
        import time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        else:
            logger.warning(f"error file exist! {log_dir}")
            logger.warning("please init new 'comment' value")
            # exit(0)
        logger.propagate = False
        log_file = os.path.join(log_dir, f"{time.strftime(format, time.localtime())}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(fm)
        logger.addHandler(file_handler)
        logger.info(f"terminal cmd: python {' '.join(sys.argv)}")
        logger.info(f"log file: {log_file}")
    else:
        logger.warning("init_logger fail")
    return logger

def select_topk(data, k=-1):
    if k<=0:
        return data
    assert k<=data.shape[1]
    val, col = torch.topk(data ,k=k)    # 返回给定输入张量沿给定维度的k个最大元素。该函数返回2个值，第一个值为排序的数组，第二个值为该数组中获取到的元素在原数组中的位置标号。
    col = col.reshape(-1)
    row = torch.ones(1, k, dtype=torch.int)*torch.arange(data.shape[0]).view(-1, 1)     # arange(): 返回一个连续的tensor; view(): the size -1 is inferred from other dimensions
    row = row.view(-1).to(device=data.device)
    new_data = torch.zeros_like(data)
    new_data[row, col] = data[row, col]
    # new_data[row, col] = 1.0
    return new_data

def RWR(A, restart_prob):
    """
    Randow Walk with Restart (RWR) on similarity network.
    :param A: n × n, similarity matrix
    :param restart_prob: probability of restart
    :return: n × n, steady-state probability
    """
    n = A.shape[0]
    A = (A + A.T) / 2
    A = A - torch.diag(torch.diag(A))     # 将对角线位置的元素置为0
    A = A + torch.diag(sum(A) == 0)
    P = A / sum(A)
    Q = torch.linalg.inv(torch.eye(n) - (1 - restart_prob) * P) @ (restart_prob * torch.eye(n))
    return Q