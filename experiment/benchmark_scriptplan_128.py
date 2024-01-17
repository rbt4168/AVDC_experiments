from mypolicy import MyPolicy
from metaworld_exp.utils import get_seg, get_cmat, collect_video, sample_n_frames
import sys
sys.path.append('core')
import imageio.v2 as imageio
import numpy as np
from myutils import get_flow_model, pred_flow_frame, get_transforms, get_transformation_matrix
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as env_dict
from metaworld import policies
from tqdm import tqdm
import cv2
import imageio
import json
import os
import random
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_policy(env_name):
    name = "".join(" ".join(env_name.split('-')[:-3]).title().split(" "))
    policy_name = "Sawyer" + name + "V2Policy"
    try:
        policy = getattr(policies, policy_name)()
    except:
        policy = None
    return policy

with open("name2maskid.json", "r") as f:
    name2maskid = json.load(f)

res2planres = {
    (640, 480): 256,
    (320, 240): 128,
    (160, 120): 64,
}

result_root = "results_scriptplan_128"
os.makedirs(result_root, exist_ok=True)
device = 'cuda:0'

n_exps = 25
resolution = (320, 240)
frames_per_plan = 8
cameras = ['corner', 'corner2', 'corner3']

model = get_flow_model()

try:
    with open(f"{result_root}/result_dict.json", "r") as f:
        result_dict = json.load(f)
except:
    result_dict = {}


for env_name in name2maskid.keys():
    if env_name in result_dict.keys():
        continue
    print(env_name)
    seg_ids = name2maskid[env_name]
    benchmark_env = env_dict[env_name]

    succes_rates = []
    reward_means, reward_stds = [], []
    gt_reward_means, gt_reward_stds = [], []
    for camera in cameras:
        success = 0
        rewards = []
        gt_rewards = []
        plan_res = 128
        for seed in tqdm(range(n_exps)):
            try: 
                env = benchmark_env(seed=seed)
                gt_policy = get_policy(env_name)
                
                obs = env.reset()
                cmat = get_cmat(env, cam_name=camera, resolution=resolution)
                seg = get_seg(env, resolution=resolution, camera=camera, seg_ids=seg_ids)
                images, depths, gt_return = collect_video(obs, env, gt_policy, camera_name=camera, resolution=resolution)
                gt_rewards.append(gt_return / len(images))
                
                depth = depths[0]
                images = np.array(sample_n_frames(images, 8)).transpose(0, 3, 1, 2)
                
                ### centercrop 128*128 on the image
                h, w = images.shape[2:]
                plan_res = res2planres[resolution]
                hr = plan_res // 2
                center_images = images[:, :, h//2-hr:h//2+hr, w//2-hr:w//2+hr].copy()
                images = np.zeros_like(images)
                images[:, :, h//2-hr:h//2+hr, w//2-hr:w//2+hr] = center_images

                ### save video plan
                os.makedirs(f'{result_root}/plans/{env_name}', exist_ok=True)
                imageio.mimsave(f'{result_root}/plans/{env_name}/{camera}_{seed}.mp4', images.transpose(0, 2, 3, 1))
                
                image1, image2, color, flow, flow_b = pred_flow_frame(model, images, device=device)
                grasp, transforms, center_2ds, sampless = get_transforms(seg, depth, cmat, flow)
                
                transform_mats = [get_transformation_matrix(*transform) for transform in transforms]
                
                env = benchmark_env(seed=seed)
                obs = env.reset()
                policy = MyPolicy(grasp, transform_mats)
                images, _, episode_return = collect_video(obs, env, policy, camera_name=camera, resolution=resolution)
                rewards.append(episode_return / len(images))
                
                ### save sample video
                os.makedirs(f'{result_root}/videos/{env_name}', exist_ok=True)
                imageio.mimsave(f'{result_root}/videos/{env_name}/{camera}_{seed}.mp4', images)
                
                print("test eplen: ", len(images))
                if len(images) <= 500:
                    success += 1
            except Exception as e:
                print(e)
                print("something went wrong, skipping this seed")
                continue
        success_rate = success / n_exps
        rewards = rewards + [0] * (n_exps - len(rewards))
        reward_means.append(np.mean(rewards))
        reward_stds.append(np.std(rewards))
        gt_rewards = gt_rewards + [0] * (n_exps - len(gt_rewards))
        gt_reward_means.append(np.mean(gt_rewards))
        gt_reward_stds.append(np.std(gt_rewards))
        
        succes_rates.append(success_rate)
                
    print(f"Success rates for {env_name}:\n", succes_rates)
    result_dict[env_name] = {
        "success_rates": succes_rates,
        "reward_means": reward_means,
        "reward_stds": reward_stds,
        "gt_reward_means": gt_reward_means,
        "gt_reward_stds": gt_reward_stds
    }
    with open(f"{result_root}/result_dict.json", "w") as f:
        json.dump(result_dict, f, indent=4)
        

    
    
    
    
    
