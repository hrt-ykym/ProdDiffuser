import torch
from diffusers import DiffusionPipeline
from PIL import Image, ImageOps
from transparent_background import Remover


def resize_with_aspect_ratio_and_padding(img, target_size):
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    delta_width = target_size[0] - img.size[0]
    delta_height = target_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def process_foreground(product_image_path, remover):
    product_image = Image.open(product_image_path).convert("RGB")
    foreground = remover.process(product_image)
    return foreground.convert("RGBA")


def combine_foreground_with_background_centered(
    foreground,
    background_image_path,
    target_size=(512, 512),
    scale=1.0,
    position=None
):
    background_image = Image.open(background_image_path).convert("RGBA")
    background_image = resize_with_aspect_ratio_and_padding(background_image, target_size)
    new_width = int(foreground.width * scale)
    new_height = int(foreground.height * scale)
    foreground = foreground.resize((new_width, new_height), Image.Resampling.LANCZOS)

    if position is None:
        x = (background_image.width - foreground.width) // 2
        y = (background_image.height - foreground.height) // 2
    else:
        x, y = position

    combined_image = background_image.copy()
    combined_image.paste(foreground, (x, y), foreground)
    return combined_image


def generate_background_with_prompt_and_mask_or_combine(
    product_image_path,
    prompt=None,
    background_image_path=None,
    seed=13,
    target_size=(512, 512),
    scale=1.0,
    position=None
):
    remover = Remover()
    foreground = process_foreground(product_image_path, remover)

    if prompt:
        model_id = "yahoo-inc/photo-background-generation"
        pipeline = DiffusionPipeline.from_pretrained(model_id, custom_pipeline=model_id)
        pipeline = pipeline.to("cuda")

        mask = ImageOps.invert(foreground.split()[-1])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.autocast("cuda"):
            generated_background = pipeline(
                prompt=prompt,
                image=foreground,
                mask_image=mask,
                control_image=mask,
                num_images_per_prompt=1,
                generator=generator,
                num_inference_steps=20,
                guess_mode=False,
                controlnet_conditioning_scale=1.0,
            ).images[0]

        return generated_background
    elif background_image_path:
        return combine_foreground_with_background_centered(
            foreground,
            background_image_path,
            target_size=target_size,
            scale=scale,
            position=position
        )
    else:
        raise ValueError("please specify prompt or background image path")
