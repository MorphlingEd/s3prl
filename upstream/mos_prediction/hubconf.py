import os

from utility.download import _urls_to_filepaths
from .expert import UpstreamExpert as _UpstreamExpert



def mos_wav2vec2_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    kwargs['upstream'] = 'wav2vec2'
    return _UpstreamExpert(ckpt, *args, **kwargs)

def mos_wav2vec2_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mos_wav2vec2_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def mos_wav2vec2_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return mos_wav2vec2_local(_urls_to_filepaths(ckpt), *args, **kwargs)

def mos_wav2vec2(refresh=False, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/s9zpouk5svu1a4l/wav2vec2-dev-SRCC-best.ckpt?dl=0'
    return mos_wav2vec2_url(refresh=refresh, *args, **kwargs)



def mos_cpc_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    kwargs['upstream'] = 'modified_cpc'
    return _UpstreamExpert(ckpt, *args, **kwargs)

def mos_cpc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mos_cpc_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def mos_cpc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return mos_cpc_local(_urls_to_filepaths(ckpt), *args, **kwargs)

def mos_cpc(refresh=False, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/qj4e44v27xf4hqc/cpc-dev-SRCC-best.ckpt?dl=0'
    return mos_cpc_url(refresh=refresh, *args, **kwargs)


def mos_tera_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    kwargs['upstream'] = 'tera'
    return _UpstreamExpert(ckpt, *args, **kwargs)

def mos_tera_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mos_tera_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def mos_tera_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return mos_tera_local(_urls_to_filepaths(ckpt), *args, **kwargs)

def mos_tera(refresh=False, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/w4jk5bujaoosk69/tera-dev-SRCC-best.ckpt?dl=0'
    return mos_tera_url(refresh=refresh, *args, **kwargs)


def mos_apc_local(ckpt, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    assert os.path.isfile(ckpt)
    kwargs['upstream'] = 'apc'
    return _UpstreamExpert(ckpt, *args, **kwargs)

def mos_apc_gdriveid(ckpt, refresh=False, *args, **kwargs):
    """
        The model from google drive id
            ckpt (str): The unique id in the google drive share link
            refresh (bool): whether to download ckpt/config again if existed
    """
    return mos_apc_local(_gdriveids_to_filepaths(ckpt, refresh=refresh), *args, **kwargs)

def mos_apc_url(ckpt, refresh=False, *args, **kwargs):
    """
        The model from URL
            ckpt (str): URL
    """
    return mos_apc_local(_urls_to_filepaths(ckpt), *args, **kwargs)

def mos_apc(refresh=False, *args, **kwargs):
    """
        The model from local ckpt
            ckpt (str): PATH
    """
    kwargs['ckpt'] = 'https://www.dropbox.com/s/ulng31as15hsvz1/apc-dev-SRCC-best.ckpt?dl=0'
    return mos_apc_url(refresh=refresh, *args, **kwargs)