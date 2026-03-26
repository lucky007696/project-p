import sys
sys.path.append('./')
from PIL import Image
import gradio as gr
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from diffusers import DDPMScheduler, AutoencoderKL
import torch
import os
import numpy as np
from utils_mask import get_mask_location
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
base_path = 'yisol/IDM-VTON'

# 1. Load Components
unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
pipe = TryonPipeline.from_pretrained(
    base_path, unet=unet, feature_extractor=CLIPImageProcessor(), torch_dtype=torch.float16
).to(device)
pipe.unet_encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16).to(device)
pipe.image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16).to(device)

# --- 4GB VRAM SURGERY ---
print("Activating 4GB VRAM safety mode...")
pipe.enable_model_cpu_offload() 
pipe.enable_vae_slicing()

# --- THE MONKEY PATCH ---
# We intercept the crashing layer. We send the data to wherever the UNet is (CPU), 
# compute it safely, and pull the result back to the GPU.
original_forward = pipe.unet.encoder_hid_proj.forward
def patched_forward(x, *args, **kwargs):
    device_orig = x.device
    dtype_orig = x.dtype
    # encoder_hid_proj weights are float16; cast x to match before forwarding
    weight_dtype = next(pipe.unet.encoder_hid_proj.parameters()).dtype
    res = original_forward(x.to(device=pipe.unet.device, dtype=weight_dtype), *args, **kwargs)
    return res.to(device=device_orig, dtype=dtype_orig)
pipe.unet.encoder_hid_proj.forward = patched_forward
# ------------------------

parsing_model = Parsing(0)
openpose_model = OpenPose(0)
tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def start_tryon(human_img_input, garm_img, garment_des, is_checked, is_checked_crop, denoise_steps, seed):
    if human_img_input is None or garm_img is None:
        return None, None
    
    if isinstance(human_img_input, dict):
        human_img_orig = human_img_input['image'].convert("RGB")
    else:
        human_img_orig = human_img_input.convert("RGB")
    
    garm_img = garm_img.convert("RGB").resize((768, 1024))
    human_img = human_img_orig.resize((768, 1024))

    print("Step 1: Detecting Pose...")
    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img_raw = args.func(args, human_img_arg)    
    pose_img_pil = Image.fromarray(pose_img_raw[:, :, ::-1]).resize((768, 1024))
    
    pose_tensor = tensor_transform(pose_img_pil).unsqueeze(0).to(device, torch.float16)
    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(device, torch.float16)

    print("Step 2: Generating Mask...")
    keypoints = openpose_model(human_img.resize((384, 512)))
    model_parse, _ = parsing_model(human_img.resize((384, 512)))
    mask, _ = get_mask_location('hd', "upper_body", model_parse, keypoints)
    mask = mask.resize((768, 1024))

    print("Step 3: Encoding Prompts & Painting... (Wait 3-5 mins)")
    steps = int(denoise_steps) if denoise_steps else 30
    generator = torch.Generator(device).manual_seed(int(seed)) if seed else None
    
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            prompt_1 = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = pipe.encode_prompt(
                prompt_1, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt
            )
            
            prompt_2 = "a photo of " + garment_des
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                [prompt_2], num_images_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=[negative_prompt]
            )

            output = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                pose_img=pose_tensor,
                image=human_img,
                mask_image=mask,
                ip_adapter_image=garm_img,
                height=1024,
                width=768,
                num_inference_steps=steps,
                guidance_scale=2.0,
                generator=generator
            )[0][0]  # <--- THIS IS THE ONLY CHANGE
            
    return output, mask
            
    return output, mask

# --- UI SETUP ---
with gr.Blocks() as demo:
    gr.Markdown("# IDM-VTON Virtual Try-On")
    with gr.Row():
        with gr.Column():
            human = gr.Image(label='Human', source='upload', type="pil", tool='sketch')
            is_checked = gr.Checkbox(label="Auto-Mask", value=True)
            crop = gr.Checkbox(label="Auto-Crop", value=False)
        with gr.Column():
            garm = gr.Image(label="Garment", source='upload', type="pil")
            desc = gr.Textbox(label="Description", value="Short sleeve black t-shirt")
        with gr.Column():
            res = gr.Image(label="Result")
            m_out = gr.Image(label="Mask View")
    
    btn = gr.Button("Run Try-On")
    btn.click(fn=start_tryon, 
              inputs=[human, garm, desc, is_checked, crop, gr.Number(value=30, visible=False), gr.Number(value=42, visible=False)], 
              outputs=[res, m_out])

demo.launch(server_name="0.0.0.0", share=True)