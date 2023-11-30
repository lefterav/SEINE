# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md


import os
import sys
import tempfile

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

from cog import BasePredictor, Input, Path
from datasets import video_transforms
from models import load_model
from models.clip import TextEmbedder
from utils import mask_generation_before


IMAGE_EXTENSIONS = [".jpeg", ".jpg", ".png", ".webp"]


def get_input(
    input_path: str, image_h: int, image_w: int, num_frames: int, mask_type: str
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
    if input_path is not None:
        print(f"loading video from {input_path}")
        if os.path.isdir(input_path):
            file_list = os.listdir(input_path)
            video_frames = []
            if mask_type.startswith("onelast"):
                num = int(mask_type.split("onelast")[-1])
                # get first and last frame
                first_frame_path = os.path.join(input_path, natsorted(file_list)[0])
                last_frame_path = os.path.join(input_path, natsorted(file_list)[-1])
                first_frame = torch.as_tensor(
                    np.array(Image.open(first_frame_path), dtype=np.uint8, copy=True)
                ).unsqueeze(0)
                last_frame = torch.as_tensor(
                    np.array(Image.open(last_frame_path), dtype=np.uint8, copy=True)
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
                    if any(
                        extention.lower().strip() == ext for ext in IMAGE_EXTENSIONS
                    ):
                        image = torch.as_tensor(
                            np.array(Image.open(file), dtype=np.uint8, copy=True)
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
                    np.array(Image.open(input_path), dtype=np.uint8, copy=True)
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


class ImageToVideoPredictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""

        args = OmegaConf.load("./configs/sample_i2v.yaml")
        self.args = args

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device

        print("Loading model...")
        model_conf = dict(
            pretrained_model_path=args.pretrained_model_path, use_mask=args.use_mask
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
        input_path: Path = Input(description="Input image."),
        prompt: str = Input(description="A description of what to generate."),
        height: int = Input(description="Height of the generated video.", default=240),
        width: int = Input(description="Width of the generated video.", default=560),
        num_frames: int = Input(
            description="The number of frames to generate.", default=16
        ),
        additional_prompt: str = Input(
            description="Additional prompt that is appended to prompt.",
            default=", slow motion.",
        ),
        sample_method: str = Input(
            description="Sampling method.", choices=["ddpm", "ddim"], default="ddpm"
        ),
        negative_prompt: str = Input(
            description="A description of what to avoid in the generation.", default=""
        ),
        guidance_scale: float = Input(default=8.0),
        seed: int = Input(
            description="Seed number for reproducability purpose.", default=-1
        ),
    ) -> Path:
        """Generate a video from image."""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        print(f"input_path        : {input_path}")
        print(f"prompt            : {prompt}")
        print(f"height            : {height}")
        print(f"width             : {width}")
        print(f"num_frames        : {num_frames}")
        print(f"additional_prompt : {additional_prompt}")
        print(f"negative_prompt   : {negative_prompt}")
        print(f"sample_method     : {sample_method}")
        print(f"guidance_scale    : {guidance_scale}")
        print(f"seed              : {seed}")

        diffusion = create_diffusion(str(self.args.num_sampling_steps))

        if seed != -1:
            print(f"Setting seed: {seed}")
            torch.manual_seed(seed)

        with torch.inference_mode():
            prompt = f"{prompt}{additional_prompt}"
            print(f"prompt: {prompt}")

            video_input, researve_frames = get_input(
                # args
                input_path=input_path,
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
            save_video_path = os.path.join(tempfile.mkdtemp(), "output.mp4")
            torchvision.io.write_video(save_video_path, video_, fps=8)

            return Path(save_video_path)
