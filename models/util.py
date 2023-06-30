import os
import imageio
import numpy as np
from typing import Union
import decord

decord.bridge.set_bridge('torch')
import torch
import torchvision
import PIL
from typing import List
from tqdm import tqdm
from einops import rearrange

from controlnet_aux import CannyDetector


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_videos_grid_pil(videos: List[PIL.Image.Image], path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def read_video(video_path, frame_rate: int = None, max_frames: int = None, max_resolution=None, start_time=0, end_time=None):
    """

    Args:
        video_path: The path to the video file to read
        frame_rate: The desired frame rate at which to read the video
        max_frames: The maximum number of frames to read from the video. Overrides end_time
        max_resolution: The maximum dimension of the height or width of the video.
        start_time: The start time in the video to begin selecting frames, in milliseconds
        end_time: The end time in the video to stop selecting frames, in milliseconds

    Returns:

    """
    # Initial video read
    vr = decord.VideoReader(video_path)
    frame_0 = vr[0].numpy()
    frame_0_image = PIL.Image.fromarray(frame_0)
    width, height = frame_0_image.size
    owidth = width
    oheight = height
    # Ensure width and height are divisible by 8
    if width % 16 != 0 or height % 16 != 0:
        # Find nearest multiple of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        print(f"\nVideo width and height must be divisible by 16, resizing from {owidth}x{oheight} to {width}x{height}")
        # Re-read the video with the adjusted width and height
        vr = decord.VideoReader(video_path, width=width, height=height)
        frame_0 = vr[0].numpy()
        frame_0_image = PIL.Image.fromarray(frame_0)
        assert frame_0_image.size == (width, height)

    video_frames = len(vr)
    video_fps = vr.get_avg_fps()
    start_frame = 0
    end_frame = None
    if end_time is not None:
        # Make sure the video is long enough to have an end time
        if end_time > video_frames / video_fps * 1000:
            print("Unable to use video end time, video is too short.")
        else:
            end_frame = int(end_time / 1000 * video_fps)

    if start_time > 0:
        start_frame = int(start_time / 1000 * video_fps)

    if end_frame is None:
        end_frame = video_frames

    if frame_rate is None:
        frame_rate = int(video_fps)

    video_length = int((end_frame - start_frame) / video_fps) * frame_rate
    print(f"Video FPS: {video_fps}, width: {width}, height: {height}, frames: {video_frames}, length: {video_length}")

    sample_index = np.linspace(start_frame, end_frame - 1, video_length, dtype=np.int32).tolist()
    if max_frames is not None:
        sample_index = sample_index[:max_frames]
        video_length = len(sample_index)

    video = vr.get_batch(sample_index)
    video = rearrange(video, "f h w c -> f c h w")
    video = (video / 127.5 - 1.0)

    return video, width, height, video_length


def get_annotation(video, annotator, annotator_args):
    t2i_transform = torchvision.transforms.ToPILImage()
    annotation = []
    for frame in tqdm(video):
        pil_frame = t2i_transform(frame)
        if isinstance(annotator, CannyDetector):
            annotator_args["low_threshold"] = 100
            annotator_args["high_threshold"] = 200
        annotation.append(annotator(pil_frame, **annotator_args))
    return annotation


# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, num_inv_steps, prompt):
    context = init_prompt(prompt, pipeline)
    uncond_embeddings, cond_embeddings = context.chunk(2)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in tqdm(range(num_inv_steps)):
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent, t, cond_embeddings, pipeline.unet)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, video_latent, num_inv_steps, prompt)
    return ddim_latents
