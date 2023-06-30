import argparse
import os
import traceback

import controlnet_aux
import cv2
import imageio
import numpy as np
import torch
from controlnet_aux import OpenposeDetector, CannyDetector, MidasDetector
from controlnet_aux.processor import MODELS, MODEL_PARAMS, Processor
from diffusers import DDIMScheduler, AutoencoderKL
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer

from models.RIFE.IFNet_HDv3 import IFNet
from models.controlnet import ControlNetModel3D
from models.pipeline_controlvideo import ControlVideoPipeline
from models.unet import UNet3DConditionModel
from models.util import save_videos_grid, read_video, get_annotation

device = "cuda"
sd_path = "checkpoints/mega_prp"
inter_path = "checkpoints/flownet.pkl"

controlnet_paths = {
    "depth": {
        "model_url": "lllyasviel/control_v11f1p_sd15_depth"
    },
    "normal": {
        "model_url": "lllyasviel/control_v11p_sd15_normalbae"
    },
    "canny": {
        "model_url": "lllyasviel/control_v11p_sd15_canny"
    },
    "mlsd": {
        "model_url": "lllyasviel/control_v11p_sd15_mlsd"
    },
    "scribble": {
        "model_url": "lllyasviel/control_v11p_sd15_scribble"
    },
    "soft_edge": {
        "model_url": "lllyasviel/control_v11p_sd15_softedge"
    },
    "segmentation": {
        "model_url": "lllyasviel/control_v11p_sd15_seg"
    },
    "openpose": {
        "model_url": "lllyasviel/control_v11p_sd15_openpose"
    },
    "lineart": {
        "model_url": "lllyasviel/control_v11p_sd15_lineart"
    },
    "anime_lineart": {
        "model_url": "lllyasviel/control_v11p_sd15s2_lineart_anime"
    }
}

controlnet_models = MODELS
controlnet_args = MODEL_PARAMS

controlnet_parser_dict = {
    "pose": OpenposeDetector,
    "depth": MidasDetector,
    "canny": CannyDetector,
}

POS_PROMPT = ""
NEG_PROMPT = "blurry, deformed, ugly, missing limbs, extra limbs, render, illustration, low quality"


def get_args():
    model_keys = list(controlnet_models.keys())
    model_string = ", ".join(model_keys)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="Text description of target video")
    parser.add_argument("--video_path", type=str, required=True, help="Path to a source video")
    parser.add_argument("--output_path", type=str, default="./outputs", help="Directory of output")
    parser.add_argument("--condition", type=str, default="depth",
                        help=f"Processor for input video. Options: {model_string}")
    parser.add_argument("--fps", type=int, default=None,
                        help="FPS of output video. If None, use the FPS of source video.")
    parser.add_argument("--max_resolution", type=int, default=None,
                        help="Maximum dimension in pixels of output video. If None, use the resolution of source video.")
    parser.add_argument("--start_time", type=int, default=0, help="Start time of output video in milliseconds")
    parser.add_argument("--end_time", type=int, default=None, help="End time of output video in milliseconds")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to generate")
    parser.add_argument("--smoother_steps", nargs='+', default=[], type=int,
                        help="Timesteps at which using interleaved-frame smoother")
    parser.add_argument("--is_long_video", action='store_true',
                        help="Whether to use hierarchical sampler to produce long video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed of generator")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    if args.condition not in controlnet_models:
        raise ValueError(f"Invalid condition {args.condition}. Options: {list(controlnet_models.keys())}")
    processor = Processor(args.condition)

    # Inspect the __call__ method of our processor, list other params
    processor_params = list(processor.__call__.__code__.co_varnames)
    # Remove the self param
    processor_params.remove("self") if "self" in processor_params else None
    print(f"Processor {args.condition} has parameters: {processor_params}")

    annotator = processor.processor
    control_model = args.condition
    if "_" in control_model:
        control_model = control_model.split("_")[0]
    model_url = controlnet_paths[control_model]["model_url"]
    tokenizer = CLIPTokenizer.from_pretrained(sd_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_path, subfolder="text_encoder").to(dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_path, subfolder="vae").to(dtype=torch.float16)
    unet = UNet3DConditionModel.from_pretrained_2d(sd_path, subfolder="unet").to(dtype=torch.float16)
    controlnet = ControlNetModel3D.from_pretrained_2d(model_url).to(dtype=torch.float16)
    interpolater = IFNet(ckpt_path=inter_path).to(dtype=torch.float16)
    scheduler = DDIMScheduler.from_pretrained(sd_path, subfolder="scheduler")

    pipe = ControlVideoPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        controlnet=controlnet, interpolater=interpolater, scheduler=scheduler,
    )
    pipe.enable_vae_slicing()
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to(device)

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    # Step 1. Read a video
    video, width, height, video_length = read_video(video_path=args.video_path, frame_rate=args.fps,
                                                    max_frames=args.max_frames, max_resolution=args.max_resolution, start_time=args.start_time,
                                                    end_time=args.end_time)
    annotator_args = controlnet_args[args.condition]

    if "detect_resolution" in processor_params and "image_resolution" in processor_params:
        # Get the max value out of width and height
        max_dim = max(width, height)
        annotator_args["detect_resolution"] = max_dim
        annotator_args["image_resolution"] = max_dim
        print(f"Using detect_resolution and image_resolution: {max_dim}")

    # Save source video
    original_pixels = rearrange(video, "(b f) c h w -> b c f h w", b=1)
    save_videos_grid(original_pixels, os.path.join(args.output_path, "source_video.mp4"), rescale=True)
    print("\nGetting annotations...")
    # Step 2. Parse a video to conditional frames
    pil_annotation = get_annotation(video, annotator, annotator_args)
    if args.condition == "depth" and controlnet_aux.__version__ == '0.0.1':
        pil_annotation = [pil_annot[0] for pil_annot in pil_annotation]

    print("\nGenerating conditioning video...")
    # Save condition video
    video_cond = [np.array(p).astype(np.uint8) for p in pil_annotation]
    imageio.mimsave(os.path.join(args.output_path, f"{args.condition}_condition.mp4"), video_cond, fps=8)

    # Reduce memory (optional)
    del annotator
    torch.cuda.empty_cache()

    # Step 3. inference
    print("\nGenerating target video...")
    if args.is_long_video:
        window_size = int(np.sqrt(video_length))
        sample = pipe.generate_long_video(args.prompt + POS_PROMPT, video_length=video_length,
                                          frames=pil_annotation,
                                          num_inference_steps=50, smooth_steps=args.smoother_steps,
                                          window_size=window_size,
                                          generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                                          width=width, height=height,
                                          ).videos
    else:
        sample = pipe(args.prompt + POS_PROMPT, video_length=video_length, frames=pil_annotation,
                      num_inference_steps=50, smooth_steps=args.smoother_steps,
                      generator=generator, guidance_scale=12.5, negative_prompt=NEG_PROMPT,
                      width=width, height=height
                      ).videos
    print("\nSaving target video...")
    save_videos_grid(sample, f"{args.output_path}/{args.prompt}.mp4")
