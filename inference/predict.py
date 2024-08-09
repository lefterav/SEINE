# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import argparse
import os
import sys
import tempfile
from operator import attrgetter

from tqdm import tqdm

try:
    import utils
    from diffusion import create_diffusion
except:
    # sys.path.append(os.getcwd())
    sys.path.append(os.path.split(sys.path[0])[0])
    # sys.path[0]
    # os.path.split(sys.path[0])
    import utils

    from diffusion import create_diffusion

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import numpy as np
import torchvision
from diffusers.models import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from natsort import natsorted
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

# from cog import BasePredictor, Input, Path
from datasets import video_transforms
from models import load_model
from models.clip import TextEmbedder
from utils import mask_generation_before


IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".webp"]


def get_input(
    image_h: int, image_w: int, num_frames: int, mask_type: str,
    file_list: list = [],
    input_path: str = None,
):
    transform_video = transforms.Compose(
        [
            video_transforms.ToTensorVideo(),  # TCHW
            video_transforms.ResizeVideo((image_h, image_w)),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    if input_path is not None or file_list is not None:
        if input_path is not None and os.path.isdir(input_path):
            file_list = os.listdir(input_path)
        elif file_list is not None:
            video_frames = []
            if mask_type.startswith("onelast"):
                num = int(mask_type.split("onelast")[-1])
                # get first and last frame
                first_frame_path = os.path.join(input_path, natsorted(file_list)[0])
                last_frame_path = os.path.join(input_path, natsorted(file_list)[-1])
                first_frame = torch.as_tensor(
                    np.array(TransitionImage.open(first_frame_path), dtype=np.uint8, copy=True)
                ).unsqueeze(0)
                last_frame = torch.as_tensor(
                    np.array(TransitionImage.open(last_frame_path), dtype=np.uint8, copy=True)
                ).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                # add zeros to frames
                num_zeros = num_frames - 2 * num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                for i in range(num):
                    video_frames.append(last_frame)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(
                    0, 3, 1, 2
                )  # f,c,h,w
                video_frames = transform_video(video_frames)
            else:
                for file in file_list:
                    # if file.endswith("jpg") or file.endswith("png"):
                    file_name, extention = os.path.splitext(file)
                    if any(
                        extention.lower().strip() == ext for ext in IMAGE_EXTENSIONS
                    ):
                        image = torch.as_tensor(
                            np.array(TransitionImage.open(file), dtype=np.uint8, copy=True)
                        ).unsqueeze(0)
                        video_frames.append(image)
                    else:
                        continue
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(
                    0, 3, 1, 2
                )  # f,c,h,w
                video_frames = transform_video(video_frames)
            return video_frames, n
        elif os.path.isfile(input_path):
            _, full_file_name = os.path.split(input_path)
            file_name, extention = os.path.splitext(full_file_name)
            # if extention == ".jpg" or extention == ".png":
            if any(extention.lower().strip() == ext for ext in IMAGE_EXTENSIONS):
                print("loading the input image")
                video_frames = []
                num = int(mask_type.split("first")[-1])
                first_frame = torch.as_tensor(
                    np.array(TransitionImage.open(input_path), dtype=np.uint8, copy=True)
                ).unsqueeze(0)
                for i in range(num):
                    video_frames.append(first_frame)
                num_zeros = num_frames - num
                for i in range(num_zeros):
                    zeros = torch.zeros_like(first_frame)
                    video_frames.append(zeros)
                n = 0
                video_frames = torch.cat(video_frames, dim=0).permute(
                    0, 3, 1, 2
                )  # f,c,h,w
                video_frames = transform_video(video_frames)
                return video_frames, n
            else:
                raise TypeError(f"{extention} is not supported !!")
        else:
            raise ValueError("Please check your path input!!")
    else:
        raise ValueError("Need to give a video or some images")


def auto_inpainting(
    # args,
    height: int,
    width: int,
    num_frames: int,
    do_classifier_free_guidance: bool,
    sample_method: str,
    video_input,
    masked_video,
    mask,
    prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    vae,
    text_encoder,
    diffusion,
    model,
    device,
    use_fp16: bool,
    use_mask: bool,
):
    b, f, c, h, w = video_input.shape
    latent_h = height // 8
    latent_w = width // 8

    # prepare inputs
    if use_fp16:
        z = torch.randn(
            1, 4, num_frames, latent_h, latent_w, dtype=torch.float16, device=device
        )  # b,c,f,h,w
        masked_video = masked_video.to(dtype=torch.float16)
        mask = mask.to(dtype=torch.float16)
    else:
        z = torch.randn(
            1, 4, num_frames, latent_h, latent_w, device=device
        )  # b,c,f,h,w

    masked_video = rearrange(masked_video, "b f c h w -> (b f) c h w").contiguous()
    masked_video = vae.encode(masked_video).latent_dist.sample().mul_(0.18215)
    masked_video = rearrange(masked_video, "(b f) c h w -> b c f h w", b=b).contiguous()
    mask = torch.nn.functional.interpolate(
        mask[:, :, 0, :], size=(latent_h, latent_w)
    ).unsqueeze(1)

    # classifier_free_guidance
    if do_classifier_free_guidance:
        masked_video = torch.cat([masked_video] * 2)
        mask = torch.cat([mask] * 2)
        z = torch.cat([z] * 2)
        prompt_all = [prompt] + [negative_prompt]

    else:
        masked_video = masked_video
        mask = mask
        z = z
        prompt_all = [prompt]

    text_prompt = text_encoder(text_prompts=prompt_all, train=False)
    model_kwargs = dict(
        encoder_hidden_states=text_prompt,
        class_labels=None,
        cfg_scale=cfg_scale,
        use_fp16=use_fp16,
    )  # tav unet

    # Sample video:
    if sample_method == "ddim":
        samples = diffusion.ddim_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            mask=mask,
            x_start=masked_video,
            use_concat=use_mask,
        )
    elif sample_method == "ddpm":
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=True,
            device=device,
            mask=mask,
            x_start=masked_video,
            use_concat=use_mask,
        )
    samples, _ = samples.chunk(2, dim=0)  # [1, 4, 16, 32, 32]
    if use_fp16:
        samples = samples.to(dtype=torch.float16)

    video_clip = samples[0].permute(1, 0, 2, 3).contiguous()  # [16, 4, 32, 32]
    video_clip = vae.decode(video_clip / 0.18215).sample  # [16, 3, 256, 256]
    return video_clip


class TransitionImage:
    def __init__(self,
                 filename):
        self.filename = filename

        # this is based on the assumption that every filename contains a numerical id
        # followed by as space, followed by a description of the picture
        filename_parts = filename.split(" ")

        # create a class variable for the id and one for the textual description
        try:
            self.file_id = filename_parts[0]
        except KeyError:
            self.file_id = 0

        try:
            self.description = " ".join(filename_parts[1:])
        except KeyError:
            self.description = filename


class ImageToVideoPredictor:
    def __init__(self, config="../configs/transition_base.yaml") -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        args = OmegaConf.load(config)
        self.args = args

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device

        print("Loading model...")
        model_conf = dict(
            pretrained_model_path=args.pretrained_model_path,
            use_mask=args.use_mask
        )
        model = load_model(**model_conf).to(device)
        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )
        # Load model weights from checkpoint
        ckpt_path = args.ckpt
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)[
            "ema"
        ]
        model.load_state_dict(state_dict)
        model.eval()

        # Create diffusion, VAE, and text encoder
        pretrained_model_path = args.pretrained_model_path
        # diffusion = create_diffusion(str(args.num_sampling_steps))
        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae").to(
            device
        )
        print("Loading text encoder...")
        text_encoder = TextEmbedder(pretrained_model_path).to(device)

        if args.use_fp16:
            print("Using half percision for inferencing!")
            vae.to(dtype=torch.float16)
            model.to(dtype=torch.float16)
            text_encoder.to(dtype=torch.float16)

        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        # self.diffusion = diffusion

    def predict(
        self,
        file_list=[], # Path = Input(description="Input image."),
        output_filename="", # = "output.mp4",
        prompt="", # = Input(description="A description of what to generate."),
        additional_prompt="", # = ", slow motion.",  # = Input(
        config="../configs/transition_inference.yaml",
        negative_prompt="",
    ):
        """Generate a video from image."""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)
        args = OmegaConf.load(config)
        height = args.height
        width = args.width
        seed = args.seed
        guidance_scale = args.guidance_scale
        num_frames = args.num_frames
        sample_method = args.sample_method

        print(f"file_list         : {file_list}")
        print(f"prompt            : {prompt}")
        print(f"height            : {height}")
        print(f"width             : {width}")
        print(f"num_frames        : {num_frames}")
        print(f"additional_prompt : {additional_prompt}")
        print(f"negative_prompt   : {negative_prompt}")
        print(f"sample_method     : {sample_method}")
        print(f"guidance_scale    : {guidance_scale}")
        print(f"seed              : {seed}")

        diffusion = create_diffusion(str(args.num_sampling_steps))

        if seed != -1:
            print(f"Setting seed: {seed}")
            torch.manual_seed(seed)

        with torch.inference_mode():
            prompt = f"{prompt}{additional_prompt}"
            print(f"prompt: {prompt}")

            video_input, reserve_frames = get_input(
                # args
                file_list=file_list,
                image_h=height,
                image_w=width,
                num_frames=num_frames,
                mask_type=self.args.mask_type,
            )  # f,c,h,w
            video_input = video_input.to(self.device).unsqueeze(0)  # b,f,c,h,w
            mask = mask_generation_before(
                self.args.mask_type, video_input.shape, video_input.dtype, self.device
            )  # b,f,c,h,w
            masked_video = video_input * (mask == 0)

            video_clip = auto_inpainting(
                height=height,
                width=width,
                num_frames=num_frames,
                do_classifier_free_guidance=self.args.do_classifier_free_guidance,
                sample_method=sample_method,
                video_input=video_input,
                masked_video=masked_video,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                cfg_scale=guidance_scale,
                vae=self.vae,
                text_encoder=self.text_encoder,
                diffusion=diffusion,
                model=self.model,
                device=self.device,
                use_fp16=self.args.use_fp16,
                use_mask=self.args.use_mask,
            )
            video_ = (
                ((video_clip * 0.5 + 0.5) * 255)
                .add_(0.5)
                .clamp_(0, 255)
                .to(dtype=torch.uint8)
                .cpu()
                .permute(0, 2, 3, 1)
            )
            # save_video_path = "/tmp/output.mp4"
            # save_video_path = os.path.join(tempfile.mkdtemp(), output_filename)
            # torchvision.io.write_video(save_video_path, video_, fps=8)
            torchvision.io.write_video(output_filename, video_, args.fps)

            return output_filename


def transition(config_base, config_inference, config_iterations):

    predictor = ImageToVideoPredictor(config_base)
    iteration_config = OmegaConf.load(config_iterations)

    # get the list of filenames of the given directory and initialize the image instances
    images = [TransitionImage(f) for f in os.listdir(iteration_config.input_directory) if os.path.isfile(f)]

    # sort the images based on their file_id and then by their description
    images = sorted(images, key=attrgetter("file_id", "description"))

    input_file_pairs = []

    if iteration_config.iteration_mode == "sequential_pairwise":
        # create pairs (1,2), (2,3)
        input_file_pairs = [(images[i], images[i+1]) for i in range(len(images) - 1)]

    for file1, file2 in tqdm(input_file_pairs):
        # create the output filename by joining the ids with a hyphen
        output_filename = os.path.join(iteration_config.output_dir, f"{file1.file_id}-{file2.file_id}.mp4")
        prompt = config_iterations.prompt.format(file1=file1, file2=file2)
        print(prompt)

        predictor.predict(config=config_inference,
                          file_list=[file1, file2],
                          output_filename=output_filename,
                          prompt=prompt,
                          )


if __name__ == '__main__':
    # read the commandline arguments specifying the config files
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config",
                        help="basic model yaml configuation file",
                        type=str,
                        default="configs/transition_base.yaml")
    parser.add_argument("--inference-config",
                        help="inference yaml configuration file",
                        type=str,
                        default="configs/transition_inference.yaml")
    parser.add_argument("--iteration-config",
                        help="configuration yaml file for an iteration",
                        type=str,
                        default="configs/transition_iterations.yaml")
    args = parser.parse_args()

    transition(args.base_config, args.inference_config, args.iteration_config)
        


