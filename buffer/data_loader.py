import os
import json
from typing import Union, Dict
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle

# GSM8K: 恢复读取 train_split / dev_split
class GSM8KLoader(Loader):
    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.strip():
                    try:
                        data = json.loads(line)
                        instance = Instance(**data)
                        ds.append(instance)
                    except:
                        pass
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_split.jsonl'),
                'dev':   os.path.join(paths, 'dev_split.jsonl'),
                'test':  os.path.join(paths, 'test_socratic.jsonl')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})

# AQuA: 保持原样 (AQuA 原本就有 train/dev/test)
class AQuALoader(Loader):
    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        instance = Instance(**data)
                        ds.append(instance)
                    except: pass
        return ds
    
    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train.json'),
                'dev':   os.path.join(paths, 'dev.json'),
                'test':  os.path.join(paths, 'test.json')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})

# DU: 保持原样 (同一个文件用于所有 split)
class DULoader(Loader):
    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path) as f:
            ins_list = json.load(f)
        for ins in ins_list:
            instance = Instance(**ins)
            ds.append(instance)
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'date_understanding_gsm_style.json'),
                'dev':   os.path.join(paths, 'date_understanding_gsm_style.json'),
                'test':  os.path.join(paths, 'date_understanding_gsm_style.json')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})

# StrategyQA: 恢复读取三个具体的 split 文件，而不是代码内切分
class StrategyQALoader(Loader):
    def _load(self, path: str) -> DataSet:
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
            
        for ins in dataset:
            ds.append(Instance(**ins))
        return ds

    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_split.json'),
                'dev':   os.path.join(paths, 'dev_split.json'),
                'test':  os.path.join(paths, 'test_split.json')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})

# AugASDivLoader: 恢复读取 train_split / dev_split
class AugASDivLoader(GSM8KLoader):
    def load(self, paths: Union[str, Dict[str, str]] = '/path/to/data/dir') -> DataBundle:
        if isinstance(paths, str):
            paths = {
                'train': os.path.join(paths, 'train_split.jsonl'),
                'dev':   os.path.join(paths, 'dev_split.jsonl'),
                'test':  os.path.join(paths, 'aug-dev.jsonl')
            }
        return DataBundle(datasets={k: self._load(v) for k, v in paths.items()})