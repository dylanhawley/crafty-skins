# Workflow: 
Create an In-Context LoRA finetune for a foundational diffusion transformer (DiT) model (e.g. Flux.1-dev). Use training-free SD-Edit pipeline with the DiT LoRA to inpaint a Minecraft skin file and its render.

## Step 1: Dataset Preparation

### Composite Image Creation  
For each data point, create a composite image with four panels:
|Panel 1  |  Panel 2  |  Panel 3  |  Panel 4  |
|----|----|----|----|
| Photorealistic Input Image  | Front Render of Target Skin  | Back Render of Target Skin  | UV Map of Target Skin (Minecraft Skin File) |

### Prompt Formatting 

Follow the [In-Context LoRA](https://ali-vilab.github.io/In-Context-LoRA-Page/) paper's approach. For each composite, generate a prompt in the format: 

> This four-panel image showcases a rustic living room with warm wood tones and cozy decor elements; `[TOP-LEFT]` features a large stone fireplace with wooden shelves filled with books and candles; `[TOP-RIGHT]` shows a vintage leather sofa draped in plaid blankets, complemented by a mix of textured cushions; `[BOTTOM-LEFT]` displays a corner with a wooden armchair beside a side table holding a steaming mug and a classic book; `[BOTTOM-RIGHT]` captures a cozy reading nook with a window seat, a soft fur throw, and decorative logs stacked neatly.

Examine whether prompts need to be customized for the skin panel. If so, use the **OpenAI API** to generate textual descriptions of each panel with `gpt-4o` model.

## Step 2: LoRA Training

**Model:** Use HuggingFace Diffusers with the **Flux1-dev** model.  
**Training:** Train a LoRA adapter using the composite images and prompts.

## Step 3: Inference & Editing

**Input:** Provide a new photorealistic image as input.  
**SD-Edit:** Use SD-Edit to mask over panels 2, 3, and 4 (renders and UV map), inpainting these regions based on the input and context.  
**Crop:** Crop out panel 4 to get the Minecraft skin (UV map).  
**Output:** Resize panel 4 crop to 64x64 px and return file.

## Notes

- The goal is to enable training-free, in-context adaptation for Minecraft skin generation.
- The composite format provides both visual context and target structure for the model.
- SD-Edit is used for flexible, mask-based inpainting during inference.