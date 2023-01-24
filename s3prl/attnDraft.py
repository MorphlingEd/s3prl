import os
from sys import getsizeof
import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from pretrain.attnDistiller.dataset import OnlineWaveDataset

from upstream.attnDistiller.model import DistillerModel, DistillerConfig
from upstream.attnDistiller.expert import UpstreamExpert as DistilExpert

from upstream.attnHuBERT.convert import load_converted_model
from upstream.attnHuBERT.expert import UpstreamExpert as HuBERTExpert


device = 'cuda'

model_dir = '/home/s1973609/repos/s3prl/s3prl/hubModels/'
dataPath = '/home/s1973609/librispeech/LibriSpeechLocal'
upstreamConfigFile = 'pretrain/attnDistiller/config_model.yaml'



print("-------------- Loading Data ------------------------------")
dataset = OnlineWaveDataset(
    task_config={'sequence_length':250000},
    bucket_size=1,
    file_path=dataPath,
    libri_root=dataPath,
    sets=['train-clean-100'],
    target_level=None,
)

dataloader = DataLoader(
    dataset,
    batch_size=1,  # for bucketing
    shuffle=False,
    num_workers=4,
    drop_last=False,
    pin_memory=True,
    collate_fn=dataset.collate_fn,
)

sample_batch = None
for i, batch in enumerate(dataloader):
    if i == 5:
        sample_batch = batch
        break
print("Sample batch Loaded!")


wave_input, wave_orig, wave_len, pad_mask = sample_batch
wave_input = wave_input.to(device)
wave_len   = wave_len.to(device)
pad_mask   = pad_mask.type(wave_input.dtype).to(device)

# wave_input - tensor with shape (batch_size, seq_len)
# wave_orig  - list of tensors, with length = batch_size and each tensor being (seq_len)
# wave_len   - tensor with shape (batch_size), each element being length
# pad_mask   - tensor with shape (batch_size, seq_len), each element being 1 or 0 (masks)
print("Shapes of elements in the sample batch:")
print(wave_input.size())
print(wave_orig[0].size())
print(wave_len.size())
print(pad_mask.size())





# print("-------------- Loading HuBERT -------------------------")
# hubert_ckpt = os.path.join(model_dir, 'hubert_converted.ckpt')
# hubert = HuBERTExpert(hubert_ckpt)
# hubert.model = hubert.model.to(device)
# print(type(hubert))


# print("-------------- Feeding Data to HuBERT ------------------------------")
# attn_selected_teacher = [4, 8, 12]
# with torch.no_grad():
#     wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
#     with torch.cuda.amp.autocast(False):
#         results = hubert(wave_orig, attn_selected_teacher)

# attnMaps = results["attention_maps"]
# print(len(attnMaps))
# for i, attnMap in enumerate(attnMaps):
#     print(f"-------------- Attention Map from HuBERT layer {i} -----------------------------")
#     print(attnMap.size())
#     print(getsizeof(attnMap))


print("-------------- Loading Distiller -------------------------")
distillerConfig = yaml.load(
                open(upstreamConfigFile, "r"), Loader=yaml.FullLoader
            )
modelConfig = DistillerConfig(distillerConfig['distiller'])
distiller = DistillerModel(modelConfig)
distiller = distiller.to(device)


print("-------------- Feeding Data to Distiller ---------------------------")
attn_selected_student = [1, 2]
with torch.no_grad():
    feat, feat_final, pred, pad_mask, attnMapsDistiller = distiller(wave_input, 
                                                pad_mask,
                                                attn_selected=attn_selected_student)

print(len(attnMapsDistiller))
for i, attnMap in enumerate(attnMapsDistiller):
    print(f"-------------- Attention Map from Distiller layer {i} -----------------------------")
    print(attnMap.size())
    print(getsizeof(attnMap))