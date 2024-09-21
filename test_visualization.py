import torch, os
from tqdm import tqdm
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from imagestone_animatediff import AnimateDiffPipeline
from diffusers.utils import export_to_gif, export_to_video
from accelerate.utils import ProjectConfiguration, set_seed
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description="Set imagestone_interval")
parser.add_argument("--start", type=int, default=0, help="Start of imagestone_interval")
parser.add_argument("--end", type=int, default=60, help="End of imagestone_interval")
args = parser.parse_args()

# Load the motion adapter
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3").to(dtype=torch.float16,device=torch.device("cuda"))
# load SD 1.5 based finetuned model
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter).to(dtype=torch.float16,device=torch.device("cuda"))
scheduler = DDIMScheduler.from_pretrained(
    model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1
)

pipe.scheduler = scheduler
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)
# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
# scheduler = DDIMScheduler.from_pretrained(
#     model_id,
#     subfolder="scheduler",
#     clip_sample=False,
#     timestep_spacing="linspace",
#     beta_schedule="linear",
#     steps_offset=1,
# )
# pipe = AnimateDiffSDXLPipeline_GN.from_pretrained(
#     model_id,
#     motion_adapter=adapter,
#     scheduler=scheduler,
#     torch_dtype=torch.float16,
#     variant="fp16",
# ).to("cuda")

# pipe = ModelScopeT2V_GN.from_pretrained("ali-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16").to(dtype=torch.float16,device=torch.device("cuda"))
# scheduler = DDIMScheduler.from_pretrained("ali-vilab/text-to-video-ms-1.7b", subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", steps_offset=1)
# pipe.scheduler = scheduler

# enable memory savings
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

set_seed(42)

prompts = [
    "Spiderman is surfing",
    "Yellow and black tropical fish dart through the sea",
    "An epic tornado attacking above a glowing city at night",
    "Slow pan upward of blazing oak fire in an indoor fireplace",
    "a cat wearing sunglasses and working as a lifeguard at pool",
    "A cybernetic samurai standing on a mountain peak, with glowing neon armor, facing a setting sun.",
    "A group of astronauts playing soccer on Mars, with the Earth visible in the background.",
    "A young boy in a red jacket flying a kite shaped like a phoenix, under a vibrant sunset.",
    "A grand piano floating in space, with stars and galaxies surrounding it, as if playing music to the cosmos.",
    "A snowy village at Christmas time, with twinkling lights, children building snowmen, and people sipping hot chocolate by the fire.",
    "A futuristic racecar speeding through a tunnel of neon lights, leaving a trail of glowing energy behind it.",
    "A young girl sitting on a windowsill, staring at a rainy cityscape while holding a steaming cup of tea.",
    "A magical forest with towering glowing mushrooms, enchanted creatures wandering between trees, and soft moonlight filtering through the branches.",
    "A pirate ship navigating stormy seas under dark, ominous clouds, with lightning illuminating the ocean.",
    "A colorful parade of exotic animals and performers winding through the streets of a vibrant city during a festival.",
    "A knight in gleaming armor standing before a massive waterfall, his sword raised high as mist swirls around him.",
    "A futuristic train racing through a desert landscape, with towering red rocks in the distance.",
    "A serene sunrise over a lake with mist rolling over the water, and a lone fisherman in a small boat.",
    "A superhero flying over a bustling city, his cape billowing in the wind as he watches over the people below.",
    "A cozy bookstore filled with floor-to-ceiling shelves, soft light coming through the windows, and a cat curled up on a chair.",
    "A surfer riding a massive wave during a golden sunset, with dolphins jumping out of the water beside him.",
    "A futuristic city built on floating islands, with flying vehicles zooming between towering skyscrapers.",
    "A group of hikers standing on a mountain summit, overlooking a vast expanse of forest and lakes below.",
    "A spaceship landing on a distant, alien planet with strange purple plants and towering rock formations.",
    "A ballroom filled with elegantly dressed people dancing under a grand chandelier, with a live orchestra playing in the background.",
    "A lone samurai standing in a bamboo forest at dusk, his katana ready as fireflies flicker around him.",
    "A dragon perched on a mountain peak, breathing fire into the sky as storm clouds gather overhead.",
    "A group of children running through a field of wildflowers, their laughter echoing through the air as a gentle breeze blows.",
    "A futuristic city on the moon, with domed habitats and vehicles designed for lunar travel.",
    "A giant robot standing in the middle of a desert, its eyes glowing as it watches the horizon.",
    "A cozy coffee shop on a rainy day, with people reading books, sipping hot drinks, and raindrops tapping against the window.",
    "A royal banquet hall filled with long tables of food, musicians playing lutes, and nobles in colorful attire.",
    "A futuristic underwater city with transparent domes, people swimming between buildings, and colorful marine life all around.",
    "A group of friends stargazing from a mountaintop, with a blanket of stars stretched out above them and the Milky Way clearly visible.",
    "A tranquil beach at sunrise, with gentle waves lapping at the shore and seagulls flying overhead.",
    "A space station orbiting a gas giant planet, with colorful nebulae and distant stars in the background.",
    "A small fishing village at dawn, with boats docked by the shore and fishermen preparing their nets.",
    "A futuristic army of robots marching through a ruined city, their eyes glowing with determination.",
    "A magical library filled with floating books, glowing scrolls, and a wise old wizard studying ancient texts.",
    "A majestic waterfall cascading into a crystal-clear lake, with vibrant rainbow reflections dancing in the mist.",
    "A bustling marketplace in a futuristic city, with vendors selling exotic alien fruits and high-tech gadgets.",
    "A peaceful meadow with a single tree, its leaves turning golden in the late afternoon sun as a gentle breeze blows.",
    "A lone astronaut standing on the surface of an alien moon, gazing at a massive ringed planet in the sky.",
    "A bustling street in a futuristic city, with people in colorful outfits walking past towering holographic advertisements.",
    "A peaceful campsite in the middle of a forest, with a glowing campfire and tents set up under the stars.",
    "A grand cathedral illuminated by candlelight, with stained glass windows casting colorful patterns on the stone floor.",
    "A young woman riding a bicycle through a park filled with cherry blossoms in full bloom, petals gently falling around her.",
    "A futuristic skyline with towering skyscrapers, flying cars zooming by, and robots walking the streets.",
    "A dragon sleeping in a cave, surrounded by piles of glittering treasure and ancient artifacts.",
    "A futuristic airport with sleek, high-tech aircraft taking off and landing, with people moving through transparent walkways.",
    "A pirate captain standing at the helm of a ship, staring into the horizon as a storm brews in the distance.",
    "A peaceful zen garden with raked sand, carefully placed stones, and a small pond with koi fish swimming in it.",
    "A futuristic laboratory filled with glowing screens, robotic arms, and scientists in sleek white coats working on advanced technology.",
    "A group of adventurers exploring an ancient temple hidden deep in the jungle, with vines hanging from crumbling stone walls.",
    "A lighthouse standing on a rocky cliff, its beam cutting through the darkness as waves crash against the shore below.",
    "A vibrant sunset over a savanna, with silhouetted elephants walking in the distance and acacia trees dotting the landscape.",
    "A futuristic city underwater, with glowing domes and schools of fish swimming through transparent tunnels.",
    "A wizard casting a powerful spell in the middle of an ancient forest, with swirling magical energy illuminating the dark surroundings.",
    "A group of space explorers discovering a massive alien artifact floating in space, with strange glowing symbols etched into its surface.",
    "A busy harbor at sunset, with ships coming and going, seagulls flying overhead, and the warm glow of lanterns reflecting on the water.",
    "A robot sitting alone on a park bench, watching the leaves fall from the trees during a peaceful autumn day.",
    "A serene mountaintop temple at sunrise, with monks meditating and a sea of clouds stretching out below.",
    "A futuristic hospital filled with advanced medical technology, doctors in sleek uniforms, and patients recovering in high-tech beds.",
    "A giant robot towering over a futuristic city, with citizens looking up in awe as it patrols the skyline.",
    "A peaceful village on the edge of a forest, with smoke rising from chimneys and children playing in the fields.",
    "A magical battle between two wizards in the middle of a storm, with lightning crackling and spells flying through the air.",
    "A group of friends sitting around a campfire on a beach at night, with the stars twinkling overhead and the sound of waves in the background.",
    "A futuristic train speeding through a neon-lit city, with towering skyscrapers and glowing billboards flashing by.",
    "A dragon soaring over a snow-covered mountain range, its wings beating powerfully against the cold wind.",
    "A serene pond in the middle of a forest, with lily pads floating on the surface and dragonflies darting through the air.",
    "A bustling market in a futuristic city, with vendors selling glowing fruits and holographic gadgets.",
    "A knight standing before a massive castle, the banners of his kingdom flapping in the wind as he prepares for battle.",
    "A futuristic spaceship flying through a vibrant nebula, with swirling clouds of gas and stars shining in the distance.",
    "A peaceful farm at sunset, with cows grazing in the fields, the farmhouse lit warmly, and a tractor parked by the barn.",
    "A massive space battle taking place above a distant planet, with ships firing lasers and explosions lighting up the darkness of space.",
    "A young girl standing in the middle of a sunflower field, the golden flowers stretching as far as the eye can see under a bright blue sky.",
    "A group of adventurers trekking through a dense jungle, their path illuminated by glowing plants and strange creatures watching from the shadows.",
    "A bustling port in a futuristic city, with massive cargo ships being loaded and unloaded by robotic cranes.",
    "A warrior standing on a cliff overlooking a vast battlefield, his armor shining in the fading light of the setting sun.",
    "A tranquil Japanese tea garden, with cherry blossoms falling into a pond and a traditional tea house nestled among the trees.",
    "A futuristic spaceport with sleek spacecraft docking and taking off, as travelers in futuristic outfits move through transparent walkways.",
    "A group of medieval knights riding through a dense forest, their armor clinking and the sound of their horses' hooves echoing through the trees.",
    "A cozy log cabin in the middle of a snowy forest, with smoke rising from the chimney and warm light glowing from the windows.",
    "A bustling city in the rain, with neon lights reflecting on wet streets and people huddling under umbrellas as they rush by.",
    "A futuristic laboratory where scientists are working on genetically engineered plants that glow in the dark and change colors.",
    "A peaceful meadow at dawn, with dew-covered grass and a lone deer grazing near a crystal-clear stream.",
    "A grand ballroom in a futuristic palace, with guests in elegant, high-tech attire dancing beneath a massive chandelier made of glowing crystals.",
    "A dragon curled up on a pile of treasure in a dark cave, its eyes glowing faintly as it guards its hoard.",
    "A space colony on a distant planet, with futuristic homes built into the rocky landscape and massive domes protecting the inhabitants from the elements.",
    "A group of pirates burying treasure on a deserted island, with palm trees swaying in the breeze and the ocean crashing against the shore.",
    "A massive skyscraper under construction in a futuristic city, with robotic workers flying between the steel beams as they assemble the building.",
    "A knight kneeling in a chapel, his sword laid before him as he prays for strength before a great battle.",
    "A futuristic park filled with holographic trees and robotic animals, with children running and playing under an artificial sky.",
    "A lone astronaut floating through space, staring at the distant Earth, with stars and galaxies all around.",
    "A magical garden filled with glowing flowers, enchanted fountains, and mythical creatures wandering among the greenery.",
    "A bustling futuristic market on an alien planet, with strange creatures selling exotic goods and glowing alien plants lining the streets.",
    "A tranquil waterfall in the middle of a dense forest, with beams of sunlight filtering through the trees and birds singing in the branches.",
    "A group of knights charging into battle, their swords raised and banners flying as they face a massive army.",
    "A futuristic city with towering skyscrapers, flying vehicles, and massive holographic advertisements lighting up the night sky.",
    "A serene mountain lake at dawn, with mist rising from the water and the reflection of snow-capped peaks mirrored on the surface.",
    "A massive robot standing in the middle of a futuristic battlefield, its eyes glowing as it prepares for combat.",
    # "A group of explorers standing before the entrance to an ancient temple hidden deep in the jungle, with vines hanging from the stone walls.",
    # "A bustling space station orbiting a distant planet, with ships coming and going and people in futuristic outfits walking through the corridors.",
    # "A peaceful village nestled in the foothills of a mountain range, with smoke rising from chimneys and children playing in the fields.",
    # "A grand wizard's tower perched on a cliff overlooking the ocean, with magical energy swirling around the spire and waves crashing below."
]

name = "imagestone"
imagestone_interval = [args.start, args.end]
for i, prompt in enumerate(prompts):
        local_path = f"./results_t2v/{name}_{imagestone_interval[0]}_{imagestone_interval[1]}/result_{i}.gif"
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        output = pipe.forward_is(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=50,
            output_type="pil",
            width=512,
            height=512,
            num_frames=16,
            imagestone_interval=imagestone_interval,
        )
        frames = output.frames[0]
        export_to_gif(frames, local_path)