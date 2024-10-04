import torch
import torchaudio

from conf import *
from util.data_loader import collate_fn

train_dataset1 = torchaudio.datasets.LIBRISPEECH(dataset_path,
                                                 url = "train-clean-100", 
                                                 download = False)
train_dataset2 = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                 url="train-clean-360", 
                                                 download = False)
train_dataset3 = torchaudio.datasets.LIBRISPEECH(dataset_path, 
                                                 url="train-other-500", 
                                                 download = True)
train_dataset = torch.utils.data.ConcatDataset([train_dataset1,train_dataset2,train_dataset3]) 

train_loader = torch.utils.data.DataLoader(train_dataset, 
                                          pin_memory = False, 
                                          batch_size = batch_size, 
                                          shuffle = shuffle, 
                                          collate_fn = collate_fn, 
                                          num_workers = num_workers)