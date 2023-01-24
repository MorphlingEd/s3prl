"""
    hubconf for Distiller
    Author: Heng-Jui Chang (https://github.com/vectominist)
"""

import os

from s3prl.util.download import _urls_to_filepaths

from .expert import UpstreamExpert as _UpstreamExpert


def attn_distiller_local(ckpt, *args, **kwargs):
    """
    The model from local ckpt
        ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    return _UpstreamExpert(ckpt, *args, **kwargs)


def attn_distiller_url(ckpt, refresh=False, *args, **kwargs):
    """
    The model from url
        ckpt (str): URL
        refresh (bool): whether to download ckpt/config again if existed
    """
    return attn_distiller_local(_urls_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)


def attn_distilhubert(refresh=False, *args, **kwargs):
    """
    DistilHuBERT
    """
    return attn_distilhubert_base(refresh=refresh, *args, **kwargs)


def attn_distilhubert_base(refresh=False, *args, **kwargs):
    """
    DistilHuBERT Base
    Default model in https://arxiv.org/abs/2110.01900
    """
    kwargs[
        "ckpt"
    ] = "https://www.dropbox.com/s/hcfczqo5ao8tul3/disilhubert_ls960_4-8-12.ckpt?dl=1"
    return attn_distiller_url(refresh=refresh, *args, **kwargs)
