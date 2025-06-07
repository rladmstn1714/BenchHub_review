---
language:
- en
license: apache-2.0
base_model:
- black-forest-labs/FLUX.1-dev
- Yuanshi/OminiControl
pipeline_tag: image-to-text
task_categories:
- image-to-text
library_name: diffusers
---

This repository contains the model described in [Image Editing As Programs with Diffusion Models](https://huggingface.co/papers/2506.04158).

Project Page: https://yujiahu1109.github.io/IEAP/
Code: https://github.com/YujiaHu1109/IEAP

## Usage

```python
from diffusers import DiffusionPipeline

# load the pretrained model
pipeline = DiffusionPipeline.from_pretrained("Cicici1109/IEAP", trust_remote_code=True)

# generate an image with the model
generated_image = pipeline(
    image_path="assets/a12.jpg",
    editing_instructions="Change the action of the woman to running and minify the woman."
)

# display the generated image
generated_image.show()
```