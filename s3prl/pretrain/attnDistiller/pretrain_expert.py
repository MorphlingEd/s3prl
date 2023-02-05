"""
    Pre-train expert for attention-based distiller
    Author: Wenxuan Li
"""

import os
import numpy as np
from easydict import EasyDict as edict
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pretrain.attnDistiller.dataset import OnlineWaveDataset
from upstream.attnDistiller.model import DistillerConfig, DistillerModel
from upstream.attnHuBERT.expert import UpstreamExpert as HuBERTExpert


def freeze_model(model):
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


class UpstreamPretrainExpert(nn.Module):
    """
    The Distiller pretrain expert
    """

    def __init__(
        self, datarc, upstream_config, device="cuda", multi_gpu=False, **kwargs
    ):
        super().__init__()

        self.datarc = datarc
        self.device = device
        self.multi_gpu = multi_gpu

        if type(upstream_config) == str:
            self.upstream_config = yaml.load(
                open(upstream_config, "r"), Loader=yaml.FullLoader
            )
            print(
                "[UpstreamPretrainExpert] - Using upstream config from:",
                upstream_config,
            )
        elif type(upstream_config) == dict:
            self.upstream_config = upstream_config
            print(
                "[UpstreamPretrainExpert] - Using upstream config from the previous experiment."
            )
        else:
            raise ValueError

        self._get_train_dataloader()
        self._get_val_dataloader()

        print("[UpstreamPretrainExpert] - Initializing model...")
        model_config = DistillerConfig(self.upstream_config["distiller"])
        self.model = DistillerForPretrain(
            model_config, edict(self.upstream_config["teacher"])
        )

        if self.multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            print(
                "[UpstreamPretrainExpert] - Multi-GPU training Enabled: "
                + str(torch.cuda.device_count())
            )
        print(
            "[UpstreamPretrainExpert] - Number of parameters: "
            + str(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        )

    def _get_train_dataloader(self):
        dataset = OnlineWaveDataset(
            self.upstream_config["task"],
            self.datarc["train_batch_size"],
            target_level=self.upstream_config["audio"]["target_level"],
            **self.datarc,
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=True,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    ##################################
    # Get the validation dataloader
    def _get_val_dataloader(self):
        dataset = OnlineWaveDataset(
            task_config=self.upstream_config['task'],
        	bucket_size = self.datarc["val_batch_size"],
        	file_path = self.datarc['file_path'],
            target_level=self.upstream_config["audio"]["target_level"],
        	sets = self.datarc['val_sets'],
        	max_timestep = self.datarc['max_timestep'],
        	libri_root = self.datarc['libri_root']
        )

        self.valDataloader = DataLoader(
            dataset,
            batch_size=1,  # for bucketing
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

    ##################################

    # Interface
    def load_model(self, all_states):
        if self.multi_gpu:
            self.model.module.distiller.load_state_dict(all_states["Distiller"])
        else:
            self.model.distiller.load_state_dict(all_states["Distiller"])

    # Interface
    def add_state_to_save(self, all_states):
        all_states["Distiller"] = (
            self.model.float().distiller.state_dict()
            if not self.multi_gpu
            else self.model.float().module.distiller.state_dict()
        )
        all_states["Config"] = self.upstream_config
        return all_states

    # Interface
    def get_train_dataloader(self):
        return self.dataloader

    def get_val_dataloader(self):
        return self.valDataloader

    # Interface
    def forward(self, data, records={}, global_step=0, log_step=1000, **kwargs):
        """
        Args:
            data:
                [wave_input, pad_mask]

            records:
                defaultdict(list), by appending contents into records,
                these contents can be averaged and logged on Tensorboard
                later by self.log_records every log_step

        Return:
            loss
        """

        wave_input, wave_orig, wave_len, pad_mask = data
        wave_input = wave_input.to(self.device)
        wave_len = wave_len.to(self.device)
        pad_mask = pad_mask.type(wave_input.dtype).to(self.device)

        loss, other_res = self.model(
            wave_input,
            wave_orig,
            wave_len,
            pad_mask,
            return_other=global_step % log_step == 0,
        )

        if global_step % log_step == 0:
            for key, value in other_res.items():
                if isinstance(value, torch.Tensor):
                    value = float(value.mean().cpu().item())
                records[key] = value

        return loss, records

    # interface
    def on_before_zero_grad(self):
        pass

    # interface
    def log_records(self, records, logger, prefix, global_step, **kwargs):
        """
        Args:
            records:
                defaultdict(list), contents already appended

            logger:
                Tensorboard SummaryWriter
                please use f'{prefix}your_content_name' as key name
                to log your customized contents

            prefix:
                used to indicate downstream and train/test on Tensorboard
                eg. 'phone/train-'

            global_step:
                global_step in runner, which is helpful for Tensorboard logging
        """
        for key, values in records.items():
            if isinstance(values, torch.Tensor) and len(values.shape) > 1:
                logger.add_image(f"{prefix}{key}", values, global_step=global_step)
            elif isinstance(values, float):
                logger.add_scalar(f"{prefix}{key}", values, global_step=global_step)


class DistillerForPretrain(nn.Module):
    """
    Distiller for pretraining
    """

    def __init__(self, config: DistillerConfig, teacher_config: edict):
        super().__init__()
        self.config = config
        self.distiller = DistillerModel(config)

        self.teacher_config = teacher_config
        # teacher = torch.hub.load("s3prl/s3prl", teacher_config.model)
        # teacher = torch.hub.load(repo_or_dir="..",model="attnhubert_base", source="local")

        teacher_model_dir = '/home/s1973609/repos/s3prl/s3prl/hubModels/'
        hubert_ckpt = os.path.join(teacher_model_dir, 'hubert_converted.ckpt')
        teacher = HuBERTExpert(hubert_ckpt)

        if (
            teacher_config.model.find("hubert") >= 0
            or teacher_config.model.find("wav2vec2") >= 0
        ):
            teacher.model.encoder.layerdrop = 0
            print("[DistillerForPretrain] - Disabled teacher's encoder layerdrop")
        assert self.distiller.n_tasks <= teacher_config.n_layers, (
            self.distiller.n_tasks,
            teacher_config.n_layers,
        )
        self.teacher = teacher
        freeze_model(self.teacher)

        print(
            "[DistillerForPretrain] - Using {} as teacher with {} layers".format(
                teacher_config.model, teacher_config.n_layers
            )
        )

        ####################################################
        # if config.loss_type == "l1":
        #     self.loss_func = nn.L1Loss(reduction="none")
        # elif config.loss_type == "l2":
        #     self.loss_func = nn.MSELoss(reduction="none")
        # else:
        #     raise NotImplementedError(config.loss_type)
        self.loss_func = nn.KLDivLoss(reduction="none")
        ####################################################

        self.cosine_loss = config.cosine_loss
        if self.cosine_loss > 0:
            print("[DistillerForPretrain] - Enabled cosine similarity loss.")

        if config.init_teacher_conv_layers:
            print(
                "[DistillerForPretrain] - "
                "Initializing feature extractor from teacher"
            )
            self.distiller.feature_extractor.load_state_dict(
                self.teacher.model.feature_extractor.state_dict()
            )
            if self.distiller.post_extract_proj is not None:
                self.distiller.post_extract_proj.load_state_dict(
                    self.teacher.model.post_extract_proj.state_dict()
                )

        if config.init_teacher_encoder_layers:
            print("[DistillerForPretrain] - " "Initializing encoder from teacher")
            self.distiller.encoder.pos_conv.load_state_dict(
                self.teacher.model.encoder.pos_conv.state_dict()
            )
            for l in range(config.encoder_layers):
                self.distiller.encoder.layers[l].load_state_dict(
                    self.teacher.model.encoder.layers[l].state_dict()
                )

    def forward(
        self,
        wave_input: torch.Tensor,
        wave_orig: list,
        wave_len: torch.Tensor,
        pad_mask: torch.Tensor,
        return_other: bool = False,
    ):
        """
        Forward function.
        Input:
            wave_input: FloatTensor (B x T_wave)
            wave_orig: List of FloatTensor
            wave_len: LongTensor (B)
            pad_mask: FloatTensor (B x T)
            return_other: Bool (returns other information for logging)
        """

        # Forward model
        student_outputs = self.distiller(wave_input, pad_mask)
        feat, feat_final, pred, pad_mask, layer_results = student_outputs
        attn_maps_student = [attn_map for _, attn_map in layer_results]
        hiddens_student   = [hidden.transpose(0, 1)   for hidden, _   in layer_results]

        with torch.no_grad():
            wave_orig = [wave.to(wave_input.device) for wave in wave_orig]
            with torch.cuda.amp.autocast(False):
                teacher_outputs = self.teacher(wave_orig, 
                                    attn_selected=self.distiller.config.attn_selected_teacher)
            attn_maps_teacher = teacher_outputs["attention_maps"]
            hiddens_teacher   = [hidden.transpose(0, 1) for hidden in teacher_outputs['layer_hiddens']]

        # # Compute all objectives
        # (
        #     total_loss,
        #     rec_loss,
        #     rec_layer_loss,
        #     feat_pen,
        #     sim_loss,
        #     sim_layer_loss,
        # ) = self.compute_loss(feat, pred, teacher_hiddens, return_other)

        # Compute all objectives
        feat_lens = torch.sum(pad_mask, dim=1)
        (
            total_loss,
            attn_loss,
            vr_loss,
            feat_loss
        ) = self.compute_attn_loss(feat, 
                                feat_lens,
                                attn_maps_student, 
                                attn_maps_teacher,
                                hiddens_student,
                                hiddens_teacher)

        if return_other:
            with torch.no_grad():
                other_res = {
                    "attn_loss": attn_loss,
                    "vr_loss":   vr_loss,
                    "feat_pen": feat_loss,
                }
        else:
            other_res = None

        return total_loss, other_res

    def compute_attn_loss(self, feat, feat_lens, attn_map_student, attn_map_teacher, hiddens_student, hiddens_teacher):
        """
        Inputs:
            attn_map_student: [(B, Num_Heads, T, T)]
            attn_map_teacher: [(...) x 3]
            hiddens_student: [(B, T, D)]
            hiddens_teacher: [(B, T, D)]
        """
        total_loss = 0

        # KL divergence between the teacher's and the student's attention maps
        ## TODO the loss should be normalized by number of **distributions**
        # kl_loss = 0
        # for S_map in attn_map_student: # S_map: B x num_heads x T x T
        #     for T_map in attn_map_teacher:
                # kl = self.loss_func(torch.log(S_map + 1e-10), T_map)
                # kl = torch.sum(kl, dim=(3, 2, 1)) 
                # kl = torch.sum(kl / feat_lens / S_map.size(0) / S_map.size(1)) # divided by seq_len, num_heads, batch_size

                # kl_loss += kl
        # kl_loss /= len(attn_map_student) * len(attn_map_teacher) # divided by number of layer pairs

        kl_loss = 0
        for S_map, T_map in zip(attn_map_student, attn_map_teacher):
            kl = self.loss_func(torch.log(S_map + 1e-10), T_map)
            kl = torch.sum(kl, dim=(3, 2, 1)) 
            kl = torch.sum(kl / feat_lens / S_map.size(0) / S_map.size(1)) # divided by seq_len, num_heads, batch_size

            kl_loss += kl
        kl_loss /= len(attn_map_student)

        # Value-Relation Loss
        # vr_loss = 0
        # for S_hidden in hiddens_student:
        #     for T_hidden in hiddens_teacher:
        #         VR_teacher = torch.softmax(torch.bmm(T_hidden, T_hidden.transpose(1, 2)) / np.sqrt(T_hidden.size()[2]), dim=2)
        #         VR_student = torch.softmax(torch.bmm(S_hidden, S_hidden.transpose(1, 2)) / np.sqrt(S_hidden.size()[2]), dim=2)
        #         vr = self.loss_func(torch.log(VR_student + 1e-10), VR_teacher)

        #         vr = torch.sum(vr, dim=(2, 1))
        #         vr = torch.sum(vr / feat_lens / VR_teacher.size(0))

        #         vr_loss += vr
        # vr_loss /= len(attn_map_student) * len(attn_map_teacher)

        vr_loss = 0
        for S_hidden, T_hidden in zip(hiddens_student, hiddens_teacher):
                VR_teacher = torch.softmax(torch.bmm(T_hidden, T_hidden.transpose(1, 2)) / np.sqrt(T_hidden.size()[2]), dim=2)
                VR_student = torch.softmax(torch.bmm(S_hidden, S_hidden.transpose(1, 2)) / np.sqrt(S_hidden.size()[2]), dim=2)
                vr = self.loss_func(torch.log(VR_student + 1e-10), VR_teacher)

                vr = torch.sum(vr, dim=(2, 1))
                vr = torch.sum(vr / feat_lens / VR_teacher.size(0))

                vr_loss += vr
        vr_loss /= len(attn_map_student)


        # Feature loss
        feat_pen = feat.float().pow(2).mean()

        total_loss = (
            kl_loss 
            + vr_loss
            + feat_pen * self.config.feat_pen_loss
        )

        return total_loss, kl_loss, vr_loss, feat_pen

    def compute_loss(self, feat, pred, target, return_other=False):
        """
        Computes loss.
        Inputs:
            feat: B x T x D
            pred: B x N x T x D
            target: B x N x T x D (originally teacher's hidden representations)
        """

        # Reconstruction loss (L1 distance in the paper)
        assert pred.shape == target.shape, (pred.shape, target.shape)
        rec_loss = self.loss_func(pred, target)  # B x N x T x D

        if return_other:
            with torch.no_grad():
                rec_layer_loss = rec_loss.mean((0, 2, 3))
        else:
            rec_layer_loss = None

        rec_loss = rec_loss.mean()

        # Cosine similarity loss
        if self.cosine_loss > 0:
            sim_loss = -F.logsigmoid(F.cosine_similarity(pred, target, dim=-1))
            # B x N x T
            if return_other:
                with torch.no_grad():
                    sim_layer_loss = sim_loss.mean((0, 2))
            else:
                sim_layer_loss = None
            sim_loss = sim_loss.mean()
        else:
            sim_loss = 0
            sim_layer_loss = None

        # Feature loss
        feat_pen = feat.float().pow(2).mean()

        total_loss = (
            rec_loss
            + feat_pen * self.config.feat_pen_loss
            + sim_loss * self.cosine_loss
        )

        return total_loss, rec_loss, rec_layer_loss, feat_pen, sim_loss, sim_layer_loss
