import os
import sys

# srcディレクトリをモジュール検索パスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.prodDiffuser import \
    generate_background_with_prompt_and_mask_or_combine


def main():
    # assets
    product_image_path = "assets/product_image.jpg"  # product image
    background_image_path = "assets/background_image.jpg"  # background image
    output_path = "output/result_image.png"  # output path
    prompt = "on the clean table"  # if you don't specify prompt, it will be None

    # parameters
    target_size = (512, 512)  # output image size
    scale = 0.3  # scale of the product image
    position = None  # position of the product image
    seed = 13  # random seed for generation
    num_inference_steps = 20  # number of inference steps
    controlnet_conditioning_scale = 1.0  # controlnet conditioning scale

    # execute
    result_image = generate_background_with_prompt_and_mask_or_combine(
        product_image_path=product_image_path,
        prompt=prompt,
        background_image_path=background_image_path,
        seed=seed,
        target_size=target_size,
        scale=scale,
        position=position,
        num_inference_steps=num_inference_steps,
        controlnet_conditioning_scale=controlnet_conditioning_scale
    )

    # output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_image.save(output_path)
    print(f"Result image saved to {output_path}")

if __name__ == "__main__":
    main()
