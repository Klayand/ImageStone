import torch, os, sys
sys.path.append("/vip_media/yetian/SST/ImageStone/") # TODO: please change this path
from tqdm import tqdm 
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from imagestone_animatediff import AnimateDiffPipeline
from diffusers.utils import export_to_gif, export_to_video
from accelerate.utils import ProjectConfiguration, set_seed
import argparse
import pandas as pd

# 解析命令行参数
parser = argparse.ArgumentParser(description="Set imagestone_interval")
parser.add_argument("--start", type=int, default=0, help="Start of imagestone_interval")
parser.add_argument("--end", type=int, default=49, help="End of imagestone_interval")
parser.add_argument(
    "--method",
    type=str,
    default="sdv1.5",
    choices=["sdv1.5"],
    help="The name of the base model to use.",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
)
parser.add_argument(
    "--pipeline",
    type=str,
    choices=["VIIV", "VIVI", "VVII", "IVVI", "IVIV", "IIVV"],
    default="VIIV",
)
args = parser.parse_args()

# 读取prompt文件
prompt_file = pd.read_csv("/vip_media/yetian/SST/ImageStone/benchmarks/Captions_ChronoMagic-Bench-150.csv", usecols=[0,1])
prompts = list(prompt_file["name"])
filenames = list(prompt_file["videoid"])
method = args.method

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=torch.float16)
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
scheduler = DDIMScheduler.from_pretrained(
    model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
)
pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

set_seed(42)

root_path = os.path.join("/vip_media/yetian/SST/ImageStone/data", f"{args.method}_{args.pipeline}_{args.start}_{args.end}", "total_150", args.tag)

if not os.path.exists(root_path):
    os.makedirs(root_path)

for filename, prompt in tqdm(zip(filenames, prompts)):
    local_path = os.path.join(root_path, filename + ".mp4")
    if not os.path.exists(local_path):

        if args.pipeline == 'VIIV':
            output = pipe.forward_VIIV(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        elif args.pipeline == 'VIVI':
            output = pipe.forward_VIVI(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        elif args.pipeline == 'VVII':
            output = pipe.forward_VVII(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        elif args.pipeline == 'IVVI':
            output = pipe.forward_IVVI(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        elif args.pipeline == 'IIVV':
            output = pipe.forward_IIVV(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        elif args.pipeline == 'IVIV':
            output = pipe.forward_IVIV(
                prompt=prompt,
                num_inference_steps=50,
                num_frames=16,
                width=512,
                height=512,
                image_stone_interval=[args.start, args.end],
            )
        frames = output.frames[0]
        export_to_video(frames, local_path)