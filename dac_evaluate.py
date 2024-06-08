import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]= '1'
from prefigure.prefigure import get_all_args, push_wandb_config
from stable_audio_tools.training import create_demo_callback_from_config
from stable_audio_tools.data.dataset import create_val_dataloader_from_config
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
import torch
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import torchaudio
import numpy as np
from audiotools import AudioSignal
import torch
import torch.nn as nn

class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals or aligned features.

    Parameters
    ----------
    scaling : int, optional
        Whether to use scale-invariant (True) or
        signal-to-noise ratio (False), by default True
    reduction : str, optional
        How to reduce across the batch (either 'mean',
        'sum', or none).], by default ' mean'
    zero_mean : int, optional
        Zero mean the references and estimates before
        computing the loss, by default True
    clip_min : int, optional
        The minimum possible loss value. Helps network
        to not focus on making already good examples better, by default None
    weight : float, optional
        Weight of this loss, defaults to 1.0.
    """

    def __init__(
        self,
        scaling: int = True,
        reduction: str = "mean",
        zero_mean: int = True,
        clip_min: int = None,
        weight: float = 1.0,
    ):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.clip_min = clip_min
        self.weight = weight
        super().__init__()

    def forward(self, x: AudioSignal, y: AudioSignal):
        eps = 1e-8
        # nb, nc, nt
        if isinstance(x, AudioSignal):
            references = x.audio_data
            estimates = y.audio_data
        else:
            references = x
            estimates = y

        nb = references.shape[0]
        references = references.reshape(nb, 1, -1).permute(0, 2, 1)
        estimates = estimates.reshape(nb, 1, -1).permute(0, 2, 1)

        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0

        _references = references - mean_reference
        _estimates = estimates - mean_estimate

        references_projection = (_references**2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps

        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling
            else 1
        )

        e_true = scale * _references
        e_res = _estimates - e_true

        signal = (e_true**2).sum(dim=1)
        noise = (e_res**2).sum(dim=1)
        sdr = 10 * torch.log10(signal / noise + eps)

        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)

        if self.reduction == "mean":
            sdr = sdr.mean()
        elif self.reduction == "sum":
            sdr = sdr.sum()
        return sdr

def compute_SNR(noise,ref):
    target = np.sum(ref**2)
    noise = np.sum(noise**2)
    return 10 * np.log10(target/noise)

def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    # Convert inputs to numpy arrays if they are lists
    if isinstance(ref, list):
        ref = np.array(ref)
    if isinstance(est, list):
        est = np.array(est)

    eps = np.finfo(ref.dtype).eps

    reference = ref.copy()
    estimate = est.copy()
    
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)


    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))

    return sisdr 

def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.
    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference
    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)
    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)
    sdr = 10. * np.log10(numerator / denominator)
    return sdr

def visqol(
    estimates: AudioSignal,
    references: AudioSignal,
    mode: str = "audio",
):  # pragma: no cover
    """ViSQOL score.

    Parameters
    ----------
    estimates : AudioSignal
        Degraded AudioSignal
    references : AudioSignal
        Reference AudioSignal
    mode : str, optional
        'audio' or 'speech', by default 'audio'

    Returns
    -------
    Tensor[float]
        ViSQOL score (MOS-LQO)
    """
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    estimates = estimates.clone().to_mono().resample(target_sr)
    references = references.clone().to_mono().resample(target_sr)

    visqols = []
    for i in range(estimates.batch_size):
        _visqol = api.Measure(
            references.audio_data[i, 0].detach().cpu().numpy().astype(float),
            estimates.audio_data[i, 0].detach().cpu().numpy().astype(float),
        )
        visqols.append(_visqol.moslqo)
    return torch.from_numpy(np.array(visqols))


if __name__ == '__main__':

    sisdrs_list=[]
    sdrs_list = []
    visqols_list=[]

    args = get_all_args()

    args.pretransform_ckpt_path = 'output/unet_train/fs2ul7m9/checkpoints/epoch=5-step=700000.ckpt'
    args.model_config ='stable_audio_tools/configs/model_configs/autoencoders/stable_audio_1_0_dac.json' 
    args.dataset_config = 'stable_audio_tools/configs/dataset_configs/local_training_example.json'

    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    with open(args.model_config) as f:
        model_config = json.load(f)

    print(model_config)

    val_dataloder = create_val_dataloader_from_config(
        dataset_config, 
        batch_size=20,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=441000,
        audio_channels=model_config.get("audio_channels", 1),
    )

    module = create_model_from_config(model_config)

    new_state_dict ={}
    #load model
    if args.pretransform_ckpt_path:
        state_dict = load_ckpt_state_dict(args.pretransform_ckpt_path)
        for key in state_dict:
            if key.startswith('autoencoder.'):
                new_key = key[12:]
                new_state_dict[new_key] = state_dict[key]
            # else:
            #     new_state_dict[key] = state_dict[key]
        # print("my new model",new_state_dict.keys())
        model_state_dict = module.state_dict()
        missing_keys = model_state_dict.keys() - new_state_dict.keys()
        # print("missing",missing_keys)
        module.load_state_dict(new_state_dict, strict = True)

    module.cuda()

    with torch.no_grad():
        for batch in tqdm(val_dataloder):
            demo_reals, info = batch
            # Remove extra dimension added by WebDataset
            if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
                demo_reals = demo_reals[0]
            encoder_input = demo_reals
            encoder_input = encoder_input.cuda()
            # if module.force_input_mono:
            #     encoder_input = encoder_input.mean(dim=1, keepdim=True)

            demo_reals = demo_reals.squeeze(1)

            latents = module.encode(encoder_input)#torch.Size([1, 64, 861])
            fakes = module.decode(latents)#
            max_length = 441000
            if fakes.shape[-1] <max_length:
                fakes = F.pad(fakes, [0, max_length - fakes.shape[-1]], "constant", 0.0)
            else:
                fakes = fakes[:,:,:max_length]
            
            fakes = fakes.squeeze(0).to(torch.float32).clamp(-1, 1).cpu()
            fakeaudio = fakes.mul(32767).to(torch.int16).cpu()

            for i in range(len(info)):
                filename = "val/recon/" + info[i]["path"].split('/')[-1]
                sourcename = "val/source" + info[i]["path"].split('/')[-1]
                torchaudio.save(filename, fakes[i], model_config["sample_rate"])
                torchaudio.save(sourcename, demo_reals[i].unsqueeze(0), model_config["sample_rate"])

                visqols = visqol(estimates=AudioSignal(filename), references=AudioSignal(sourcename).zero_pad_to(441000)).cpu().item()
                # visqols = 1
                SISDR = calculate_sisdr(ref=demo_reals[i].cpu().numpy(), est=fakes[i].cpu().numpy())
                # if SISDR>20:
                #     torchaudio.save("val/20dB/speech/" + info[i]["path"].split('/')[-1], fakes[:,:AudioSignal("val/speech/" + info[0]["path"].split('/')[-1]).audio_data.shape[-1]], model_config["sample_rate"])
                #     torchaudio.save("val/20dB/speech/" +'recon_'+ info[i]["path"].split('/')[-1], demo_reals[:,:AudioSignal("val/speech/" + info[0]["path"].split('/')[-1]).audio_data.shape[-1]], model_config["sample_rate"])
                SDR = calculate_sdr(ref=demo_reals[i].cpu().numpy(), est=fakes[i].cpu().numpy())
                print(filename," SDR: {:.3f}, SISDR: {:.3f}, visqol: {:.3f}".format(SDR, SISDR, visqols))
                sisdrs_list.append(SISDR)
                sdrs_list.append(SDR)
                visqols_list.append(visqols)

        mean_sisdr = np.mean(sisdrs_list)
        mean_sdr = np.mean(sdrs_list)
        mean_visqol = np.mean(visqols_list)
        print("mean SDR: {:.3f}, mean SISDR: {:.3f}, mean visqols: {:.3f}".format(mean_sdr, mean_sisdr, mean_visqol))


