import argparse, os, sys, glob
import torch
import numpy as np
import nibabel as nib
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from main import DataModuleFromConfig

from monai import transforms
from monai.data import Dataset as MonaiDataset

keys = ["PD", "PD_FS", ]
crop_size = (160, 160, 128)

def get_fastmri_dataset(data_path, subject_keys):
    
    data = {}
    for key in subject_keys:
        sub_path = os.path.join(data_path, f"{os.path.basename(data_path)}_{key}.nii.gz")
        data[key] = sub_path

    fastmri_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=subject_keys, allow_missing_keys=True),
            transforms.EnsureChannelFirstd(keys=subject_keys, allow_missing_keys=True),
            transforms.Lambdad(keys=subject_keys, func=lambda x: x[0, :, :, :]),
            transforms.AddChanneld(keys=subject_keys),
            transforms.EnsureTyped(keys=subject_keys),
            transforms.Orientationd(keys=subject_keys, axcodes="RAI", allow_missing_keys=True),
            transforms.SpatialPadd(keys=subject_keys, spatial_size=crop_size, allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=subject_keys, lower=0, upper=99.75, b_min=0, b_max=1),
        ]
    )
        
    return MonaiDataset(data=[data], transform=fastmri_transforms)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def save_nifti(img, path):
    if len(img.shape) != 4: return
    img = img.permute(1, 2, 3, 0)  # reorder dimensions to be compatible with nibabel

    img = img.numpy()

    os.makedirs(os.path.split(path)[0], exist_ok=True)

    nifti_img = nib.Nifti1Image(img, np.eye(4))  # you might want to replace np.eye(4) with the correct affine matrix
    nib.save(nifti_img, path)

def sample_fastmri(opt, model, source):
    x_src = source.unsqueeze(0).to(device)
    z_src , _, _ = model.first_stage_model.encode(x_src)
    z_tgtl, _, _ = model.first_stage_model.encode(x_src, opt.target)
    z_src = model.get_first_stage_encoding(z_src).detach()
    z_tgtl = model.get_first_stage_encoding(z_tgtl).detach()

    z_src = torch.cat([z_src, z_tgtl], dim=1)

    x0 = z_src
    c = modalities.index(opt.target)
    c = torch.nn.functional.one_hot(torch.tensor(c), num_classes=2).float()
    c = c.unsqueeze(0).repeat(z_src.shape[0], 1).unsqueeze(1).to(device)
    shape = z_src.shape[1:]
    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                        conditioning=c,
                                        batch_size=opt.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        x0=x0,
                                        eta=opt.ddim_eta)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

    rec_src = model.decode_first_stage(z_src[:,:z_src.shape[1]//2])
    rec_src = torch.clamp((rec_src+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

    tgtl = model.decode_first_stage(z_tgtl)
    tgtl = torch.clamp((tgtl+1.0)/2.0, min=0.0, max=1.0).detach().cpu()

    return x_samples_ddim, rec_src, tgtl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--base",
        type=str,
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )

    parser.add_argument(
        "--datadir",
        type=str,
        nargs="?",
        help="dir of input data",
        default="../../../data/public_data/fastMRI/knee_mri_clinical_seq_converted_selected_corrected_split_resampled/test"
    )

    parser.add_argument(
        "--source",
        type=str,
        nargs="?",
        default="PD",
        help="the source modality (select from PD, PD_FS)",
    )

    parser.add_argument(
        "--target",
        type=str,
        nargs="?",
        default="PD_FS",
        help="the target modality (select from PD, PD_FS)",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    config = OmegaConf.load(opt.base)  # TODO: Optionally download from same location as ckpt and chnage this logic
    
    model = load_model_from_config(config, config.model.params.ckpt_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    modalities = ['PD', 'PD_FS']

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_subjects = sorted(glob.glob(os.path.join(opt.datadir, "*")))
    
    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            for subject in tqdm(all_subjects, desc="Data"):
                monai_dataset = get_fastmri_dataset(subject, [opt.source])
                subject_id = os.path.basename(subject)

                # sliding window inference, ensure the target shape is correct
                source_data = monai_dataset[0][opt.source]
                x_samples_ddim = torch.zeros_like(source_data)  # Initialize with zeros
                rec_src = torch.zeros_like(source_data)
                tgtl =  torch.zeros_like(source_data)
                weight_map = torch.zeros_like(source_data)
                
                for i in range(0, source_data.shape[1], 48):
                    for j in range(0, source_data.shape[2], 48):
                        for k in range(0, source_data.shape[3], 64):
                            batch = source_data[:, i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]]
                            
                            if batch.shape[1:] != crop_size:
                                continue
                            each_x_samples_ddim, each_rec_src, each_tgtl = sample_fastmri(opt, model, batch)
                            x_samples_ddim[:, i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]] += each_x_samples_ddim.squeeze(0)
                            rec_src[:, i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]] += each_rec_src.squeeze(0)
                            tgtl[:, i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]] += each_tgtl.squeeze(0)
                            weight_map[:, i:i+crop_size[0], j:j+crop_size[1], k:k+crop_size[2]] += 1
            
                for i in range(0, source_data.shape[1], 48):
                    for k in range(0, source_data.shape[3], 64):
                        batch = source_data[:, i:i+crop_size[0], source_data.shape[2]-crop_size[1]:source_data.shape[2], k:k+crop_size[2]]
                            
                        if batch.shape[1:] != crop_size:
                            continue
                        each_x_samples_ddim, each_rec_src, each_tgtl = sample_fastmri(opt, model, batch)
                        x_samples_ddim[:, i:i+crop_size[0], source_data.shape[2]-crop_size[1]:source_data.shape[2], k:k+crop_size[2]] += each_x_samples_ddim.squeeze(0)
                        rec_src[:, i:i+crop_size[0], source_data.shape[2]-crop_size[1]:source_data.shape[2], k:k+crop_size[2]] += each_rec_src.squeeze(0)
                        tgtl[:, i:i+crop_size[0], source_data.shape[2]-crop_size[1]:source_data.shape[2], k:k+crop_size[2]] += each_tgtl.squeeze(0)
                        weight_map[:, i:i+crop_size[0], source_data.shape[2]-crop_size[1]:source_data.shape[2], k:k+crop_size[2]] += 1
                
                
                weight_map[weight_map == 0] = 1  # Avoid division by zero
                x_samples_ddim = x_samples_ddim / weight_map
                rec_src = rec_src / weight_map
                tgtl = tgtl / weight_map
                # for x_sample in x_samples_ddim:
                #     save_nifti(x_sample, os.path.join(sample_path, os.path.join( f"{base_count:04}.nii.gz")))
                #     base_count += 1
                
                save_nifti(x_samples_ddim, os.path.join(sample_path, f"{subject_id}_{opt.source}_to_{opt.target}.nii.gz"))
                save_nifti(rec_src, os.path.join(sample_path, f"{subject_id}_{opt.source}_reconstruction.nii.gz"))
                save_nifti(tgtl, os.path.join(sample_path, f"{subject_id}_{opt.target}_reconstruction.nii.gz"))

    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")