import os

from prodDiffuser import generate_background_with_prompt_and_mask_or_combine

# assets
product_image_path = "assets/product_image.jpg"  # product image
background_image_path = "assets/background_image.jpg"  # background image
output_path = "output/result_image.png"  # output path
prompt = "on the clean table"  # if you don't specify prompt, it will be None

# execute
result_image = generate_background_with_prompt_and_mask_or_combine(
    product_image_path=product_image_path,
    prompt=prompt,
    background_image_path=background_image_path,
    scale=0.3,
    position=None
)

# output
os.makedirs(os.path.dirname(output_path), exist_ok=True)
result_image.save(output_path)
print(f"Result image saved to {output_path}")
