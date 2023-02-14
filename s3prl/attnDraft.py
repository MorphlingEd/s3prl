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
import torch.nn.functional as F

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
batch_size = 4
dataset = OnlineWaveDataset(
    task_config={'sequence_length':250000},
    bucket_size=batch_size,
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

# num_params = sum(p.numel() for p in hubert.model.parameters())
# print(f"Number of parameters in HuBERT: {num_params}")


print("-------------- Loading Distiller -------------------------")
distillerConfig = yaml.load(
                open(upstreamConfigFile, "r"), Loader=yaml.FullLoader
            )
modelConfig = DistillerConfig(distillerConfig['distiller'])
distiller = DistillerModel(modelConfig)
distiller = distiller.to(device)

num_params = sum(p.numel() for p in distiller.parameters())
print(f"Number of parameters in Distiller: {num_params}")


# print("Initializing distiller from the teacher...")
# distiller.encoder.pos_conv.load_state_dict(
#     hubert.model.encoder.pos_conv.state_dict()
# )
# for l in range(2):
#     distiller.encoder.layers[l].load_state_dict(
#         hubert.model.encoder.layers[l].state_dict()
#     )

# print("Saving...")
# all_states = {}
# all_states["Distiller"] = distiller.state_dict()
# all_states["Config"] = distillerConfig

# save_path = os.path.join(model_dir, "baseline.ckpt")
# torch.save(all_states, save_path)


# print("-------------- Feeding Data to HuBERT ------------------------------")
# with torch.no_grad():
#     wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
#     with torch.cuda.amp.autocast(False):
#         results = hubert(wave_orig, distiller.config.attn_selected_teacher)

# attnMapsTeacher = results["attention_maps"]
# hiddensTeacher  = [hidden.transpose(0, 1) for hidden in results['layer_hiddens']]
# print(len(attnMapsTeacher))
# for i, attnMap in zip(distiller.config.attn_selected_teacher, attnMapsTeacher):
#     print(f"-------------- Attention Map from HuBERT layer {i} -----------------------------")
#     print(attnMap.size())
#     print(getsizeof(attnMap))

# print(len(hiddensTeacher))
# for i, hidden in zip(distiller.config.attn_selected_teacher, hiddensTeacher):
#     print(f"-------------- Hidden representations from Teacher layer {i} --------------------")
#     print(hidden.size())
#     print(getsizeof(hidden))

# print('---------------- Size of teacher\'s final outputs -----------------')
# pred_teacher = results["last_hidden"]
# print(pred_teacher.size())
# print(torch.transpose(pred_teacher, 0, 1).size())



print("-------------- Feeding Data to Distiller ---------------------------")
attn_selected_student = [2]
with torch.no_grad():
    feat, feat_final, pad_mask, layerResults = distiller(wave_input, pad_mask)
    attnMapsDistiller = [attnMap for _, attnMap in layerResults]
    hiddensDistiller  = [hidden.transpose(0, 1) for hidden, _   in layerResults]

# print(len(attnMapsDistiller))
# for i, attnMap in zip(attn_selected_student ,attnMapsDistiller):
#     print(f"-------------- Attention Map from Distiller layer {i} -----------------------------")
#     print(attnMap.size())
#     print(getsizeof(attnMap))

# print(len(hiddensDistiller))
# for i, hidden in zip(attn_selected_student ,hiddensDistiller):
#     print(f"-------------- Hidden representations from Distiller layer {i} --------------------")
#     print(hidden.size())
#     print(getsizeof(hidden))



print('---------------- Size of student\'s outputs -------------------------')
print(f"feat: {feat.size()}") # B x T x feat_dim (after convolutional layers)
print(f"feat_final: {feat_final.size()}") # B x T x hidden_dim (D)
# print(f"pred: {pred.size()}") # B x N x T x hidden_dim (but not sure what N is)


# print('------------------------- Pad Mask ----------------------------------')
# print(f"pad_mask: {pad_mask.size()}")
# print(f"pad_mask requires_grad: {pad_mask.requires_grad}")
# print(pad_mask)




# kl_loss = torch.nn.KLDivLoss(reduction='none')
# loss = 0

# feat_lens = torch.sum(pad_mask, dim=1)
# print(f"feat_lens: {feat_lens}")


# print('-------------------- Calculating Attention-based loss ---------------------------')
# for i, attn_teacher in zip(distiller.config.attn_selected_teacher, attnMapsTeacher):
#     for j, attn_student in zip(attn_selected_student, attnMapsDistiller):
#         kl = kl_loss(torch.log(attn_student + 1e-10), attn_teacher)

#         # Normalization: divided by batch_size, number of heads, sequence lengths and later number of layer pairs
#         kl = torch.sum(kl, dim=(3, 2, 1)) # sum over num_heads, 
#         kl = torch.sum(kl / feat_lens / attn_student.size(0) / attn_student.size(1)) 

#         print(f"KL divergence between attention maps from student layer {j} and teacher layer {i} is: {kl}")
#         loss += kl

# loss /= len(attnMapsTeacher) * len(attnMapsDistiller)
# print(f"average Attn loss: {loss}")


# print('-------------------- Calculating Value Relation loss ---------------------------')
# loss = 0
# for i, T_hidden in zip(distiller.config.attn_selected_teacher, hiddensTeacher):
#     for j, S_hidden in zip(attn_selected_student, hiddensDistiller):
#         VRTeacher = torch.softmax(torch.bmm(T_hidden, T_hidden.transpose(1, 2)) / np.sqrt(T_hidden.size()[2]), dim=2)
#         VRStudent = torch.softmax(torch.bmm(S_hidden, S_hidden.transpose(1, 2)) / np.sqrt(S_hidden.size()[2]), dim=2)
#         vr = kl_loss(torch.log(VRStudent + 1e-10), VRTeacher)

#         # Normalization: divided by batch_size, sequence lengths and later number of layer pairs
#         vr = torch.sum(vr, dim=(2, 1)) 
#         vr = torch.sum(vr / feat_lens / T_hidden.size(0)) 

#         print(f"KL divergence between value relation from student layer {j} and from teacher layer {i} is: {vr}")
#         loss += vr

# loss /= len(hiddensTeacher) * len(hiddensDistiller)
# print(f"average VR loss: {loss}")