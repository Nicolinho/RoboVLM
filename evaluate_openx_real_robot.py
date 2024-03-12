import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from palme_model_openx import Palme
from accelerate import Accelerator
from os.path import join as pjoin
from tqdm import tqdm


from pathlib import Path
import os
# rtx relevant import
# from tf_agents.trajectories import time_step as ts
# from tf_agents.policies import py_tf_eager_policy


# import tensorflow_hub as hub
import torch
import hydra
import cv2
from datetime import datetime
import gym
from typing import Any
from robot_io.envs.robot_env import RobotEnv

# from hulc2.evaluation.utils import imshow_tensor
# # from hulc2.models.hulc2 import Hulc2
# # from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs

# from calvin_env.utils.utils import angle_between_angles
from robot_io.utils.utils import quat_to_euler


def angle_between_angles(a, b):
    diff = b - a
    return (diff + np.pi) % (2 * np.pi) - np.pi


def obs_dict_to_np(robot_obs):
    tcp_pos = robot_obs["tcp_pos"]
    tcp_orn = quat_to_euler(robot_obs["tcp_orn"])
    gripper_width = robot_obs["gripper_opening_width"]
    gripper_action = 1 if gripper_width > 0.06 else -1

    return np.concatenate([tcp_pos, tcp_orn, [gripper_action]])


class PandaRTXWrapper(gym.Wrapper):
    """
    Compared to PandaLfpWrapper, this wrapper doesn't require dataset input, doesn't apply transform
    to observation return
    """

    def __init__(
        self,
        env: RobotEnv,
        relative_action: bool = True,
        device: str = "cuda:0",
        max_rel_pos: float = 0.02,
        max_rel_orn: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super(PandaRTXWrapper, self).__init__(env)
        self.env = env
        self.max_rel_pos = max_rel_pos
        self.max_rel_orn = max_rel_orn
        self.device = device
        self.relative_actions = relative_action
        # logger.info(f"Initialized PandaRTXWrapper for device {self.device}")
        # logger.info(f"Relative actions: {self.relative_actions}")

    def step(self, action_tensor):
        if self.relative_actions:
            action_tensor = torch.clamp(action_tensor, -1, 1)
        action = np.split(
            action_tensor.squeeze().cpu().detach().numpy(), [3, 6])
        if self.relative_actions:
            # scale actions to metric values
            action[0] *= self.max_rel_pos
            action[1] *= self.max_rel_orn
        action[2] = 1 if action[-1] > 0 else -1
        action_dict = {"motion": action,
                       "ref": "rel" if self.relative_actions else "abs"}
        o, r, d, i = self.env.step(action_dict)

        # obs = self.transform_observation(o)
        obs = o
        return obs, r, d, i

    def reset(self, episode=None, robot_obs=None, target_pos=None, target_orn=None, gripper_state="open"):
        if episode is not None:
            robot_obs = episode["state_info"]["robot_obs"][0]

        if robot_obs is not None:
            robot_obs = robot_obs.cpu().numpy()
            target_pos = robot_obs[:3]
            target_orn = robot_obs[3:6]
            gripper_state = "open" if robot_obs[-1] == 1 else "closed"
            obs = self.env.reset(
                target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        elif target_pos is not None and target_orn is not None:
            obs = self.env.reset(
                target_pos=target_pos, target_orn=target_orn, gripper_state=gripper_state)
        else:
            obs = self.env.reset()

        # return self.transform_observation(obs)
        return obs

    def get_obs(self):
        obs = self.env._get_obs()
        # return self.transform_observation(obs)
        return obs


def lang_rollout(model, env, goal, hist_len, ep_len=500, gt_traj=None):
    print("Type your instruction which the robot will try to follow")
    # while 1:
    #     lang_input = [input("What should I do? \n")]
    #     goal = lang_input[0]
    #     print("sleeping 5 seconds...)")
    #     time.sleep(6)
    #     rollout(env, model, goal, embed_model)
    print(goal)
    rollout(env, model, goal, hist_len, ep_len=ep_len, gt_traj=gt_traj)


def to_relative_action(actions, robot_obs, max_pos=0.02, max_orn=0.05):
    assert isinstance(actions, np.ndarray)
    assert isinstance(robot_obs, np.ndarray)
    # assert isinstance(actions, torch.tensor)
    # assert isinstance(robot_obs, torch.tensor)

    rel_pos = actions[:3] - robot_obs[:3]
    rel_pos = np.clip(rel_pos, -max_pos, max_pos) / max_pos

    rel_orn = angle_between_angles(robot_obs[3:6], actions[3:6])
    rel_orn = np.clip(rel_orn, -max_orn, max_orn) / max_orn

    gripper = actions[-1:]
    return np.concatenate([rel_pos, rel_orn, gripper])


def _unscale_actions_by_bounds(actions, lows, highs, safety_margin=0.01):
    return (actions + 1) * (highs - lows) / 2 + lows


def _unscale_action(action):
    """Rescales actions based on measured per dimension ranges."""
    # Rotation Delta
    # rd_lows = tf.constant([-3.2, -0.8, -1.8])
    # rd_highs = tf.constant([3.2, 0.2, 2.5])
    # action['rotation_delta'] = _unscale_actions_by_bounds(
    #     action['rotation_delta'], lows=rd_lows, highs=rd_highs
    # )
    #
    # # World Vector
    # wv_lows = tf.constant([0.0, -0.5, 0.0])
    # wv_highs = tf.constant([0.8, 0.7, 0.6])
    # action['world_vector'] = _unscale_actions_by_bounds(
    #     action['world_vector'], lows=wv_lows, highs=wv_highs
    # )

    lows = np.array([0.0, -0.5, 0.0, -3.2, -0.8, -1.8])
    highs = np.array([0.8, 0.7, 0.6, 3.2, 0.2, 2.5])

    action[:6] = (action[:6] + 1) * (highs - lows) / 2 + lows

    return action


def rollout(env, model, goal, hist_len, ep_len=5000, gt_traj=None):
    # env.reset()
    obs = env.get_obs()
    # obs = {'rgb_static': np.random.randint(0, 255, size = (200,200,3), dtype=np.uint8)}

    if model is not None:
        model.reset_history(instruction=goal, max_len=hist_len)
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # # dd/mm/YY H:M:S
    # goal_str = "_".join(goal.split()) + "_imgs"
    # dt_string = now.strftime("%Y_%m_%d_at_%H_%M_%S")
    # folder = Path("/tmp") / goal_str / dt_string
    # os.makedirs(folder, exist_ok=True)
    pbar = tqdm(range(ep_len), total=ep_len)
    for step in pbar:
        pbar.set_description(f"Step {step + 1}/{ep_len}")

        if gt_traj is not None:
            gt_action = gt_traj['actions_unprocessed'][step]

        if model is not None:
            pred_action = model.select_action(obs['rgb_static'])

            if gt_traj is not None:
                print("GT vs. Pred differences: ", np.array(
                    gt_action) - np.array(pred_action))

        action = pred_action

        # print(action)
        # now = datetime.now()
        action = _unscale_action(np.array(action))

        curr_pose = obs_dict_to_np(obs["robot_state"])
        # TODO can this be done in torch?
        rel_act = to_relative_action(action, curr_pose)
        rel_act_torch = torch.tensor(rel_act)

        # rel_act_torch = torch.tensor(action)

        obs, _, _, _ = env.step(rel_act_torch)

        cv2.imshow("rgb_static", obs["rgb_static"][:, :, ::-1])
        # save_path = folder / f"{step:03}.png"
        # cv2.imwrite(save_path.as_posix(), obs["rgb_static"][:, :, ::-1])

        k = cv2.waitKey(1)

        # k = imshow_tensor("rgb_static", obs["rgb_static"], wait=1, resize=True, text=goal)
        # press ESC to stop rollout and return
        # if k == 27:
        #     return


@hydra.main(config_path="../config", config_name="inference_real_rtx")
def main(cfg):
    # load robot
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    env = PandaRTXWrapper(env, relative_action=True)

    # robot, env = None, None

    # scp -r taco_alldata_vitL224_norm_noquant/checkpoint-2642/dequant dorka@knoppers:~/chkpts/taco_alldata_vitL224_norm_noquant/checkpoint-2642
    # scp -r taco_extradata_vitL_noquant/checkpoint-435/dequant dorka@knoppers:~/chkpts/taco_extradata_vitL_noquant/checkpoint-435
    # scp -r taco_extradata_clipL_single_proj/checkpoint-703/dequant/ dorka@knoppers:~/chkpts/taco_extradata_clipL_single_proj/checkpoint-703
    # scp -r taco_extradata_vitL/checkpoint-352/dequant dorka@knoppers:~/chkpts/taco_extradata_vitL/checkpoint-352

    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_vitL/checkpoint-352"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_clipL_single_proj/checkpoint-703"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_vitL_noquant/checkpoint-435"
    checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_qwen/checkpoint-435"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_llava7bv15_tune_proj_lm/checkpoint-215"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_llava7bv15_tune_lm/checkpoint-215"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_llava7bv15_tune_proj/checkpoint-215"
    # checkpoint_dir_path = "/export/home/dorka/chkpts/taco_extradata_instrblib7b/checkpoint-351"

    # checkpoint_dir_path = "/home/dorka/chkpts/taco_alldata/checkpoint-876"

    acces_token = "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"
    llama_checkpoint = "meta-llama/Llama-2-7b-hf"

    if 'qwen' in checkpoint_dir_path:
        checkpoint_image_model = 'Qwen/Qwen-VL'
    elif 'blib' in checkpoint_dir_path:
        checkpoint_image_model = "Salesforce/instructblip-vicuna-7b"
    elif 'llava' in checkpoint_dir_path:
        checkpoint_image_model = "llava-hf/llava-1.5-7b-hf"
    elif 'clipL' in checkpoint_dir_path:
        checkpoint_image_model = "openai/clip-vit-large-patch14"
    elif 'vitL' in checkpoint_dir_path:
        checkpoint_image_model = "google/vit-large-patch16-224"

    device_index = Accelerator().process_index
    device_map = {"": device_index}

    # for inference, this can be adjusted during inference. Differnt values could be tried, but 5 is maybe good tradeoff between speed and performance
    # models are trained with hist len up to 10
    hist_len = 5

    # model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
    #               config=None, output_dir = None,
    #               load_in_8bit=True,
    #               lora_lm=True, lora_vision=False, freeze_vision=True,
    #               device_map=device_map,
    #               torch_dtype = torch.bfloat16,
    #               # torch_dtype = torch.float16,
    #               )

    dummy_model = False

    if dummy_model:
        model = None
    else:
        model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token,
                      image_model_name=checkpoint_image_model,
                      load_in_8bit=False, load_in_4bit=False,
                      lora_lm=False, lora_vision=False, freeze_vision=True,
                      quantize_vision=False,
                      device_map=device_map,
                      torch_dtype=torch.bfloat16,
                      # torch_dtype = torch.float16,
                      #   flash_attn="flash_attention_2",
                      # flash_attn = "sdpa",
                      )

        if 'openai/clip' in checkpoint_image_model or 'google/vit' in checkpoint_image_model:
            print("load trained model")
            model.lm.load_state_dict(torch.load(
                pjoin(checkpoint_dir_path, 'dequant', 'lm_model.bin'), map_location="cpu"))

        if 'openai/clip' in checkpoint_image_model:
            model.proj_layer.load_state_dict(torch.load(
                pjoin(checkpoint_dir_path, 'dequant', 'img_proj_layer_model.bin'), map_location="cpu"))
        elif 'google/vit' in checkpoint_image_model:
            model.img_embed_model.classifier.load_state_dict(torch.load(
                pjoin(checkpoint_dir_path, 'dequant', 'img_embed_model_classifier_model.bin'), map_location="cpu"))

        if any([n in checkpoint_image_model for n in ['Qwen', 'llava', 'blip']]):
            model.load_state_dict(
                torch.load(pjoin(checkpoint_dir_path, "pytorch_model_merged.bin"), map_location="cpu"))

        model.do_torch_compile()
        # save_pretrained to be able to load it faster

    # model.load("/home/dorka/projects/23/llama-openx/palme/palme/taco_alldata/checkpoint-880/pytorch_model.bin")
    # model.load("/home/dorka/chkpts/run_openx_test/checkpoint-173/pytorch_model.bin")
    #
    # model.lm = model.lm._unload_and_optionally_merge(dtype=torch.bfloat16) # does not work on titan x
    # ## model = model.merge_and_unload() # does not work on titan x
    #
    # torch.save(model.lm.state_dict(),
    #            "/home/dorka/chkpts/run_openx_test/checkpoint-173/dequant_lm2/pytorch_model.bin")
    # torch.save(model.img_embed_model.classifier.state_dict(),
    #            "/home/dorka/chkpts/run_openx_test/checkpoint-173/dequant_lm2/img_embed_model_classifier_model.bin")

    # model.lm.load_state_dict(torch.load(
    #     "/home/dorka/chkpts/run_openx_test/checkpoint-173/dequant_lm2/pytorch_model.bin",
    #                                    map_location="cpu"))
    #
    # model.img_embed_model.classifier.load_state_dict(torch.load(
    #     "/home/dorka/chkpts/run_openx_test/checkpoint-173/dequant_lm2/img_embed_model_classifier_model.bin",
    #                                    map_location="cpu"))

    load_demonstration = False
    if load_demonstration:
        print("Load dataset")
        from dataset_tools_openx import generator_taco_extra_data
        traj_list = [traj for traj in generator_taco_extra_data(
            # data_path="/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
            data_path="/export/home/huang/taco_extra_processed_15hz_resize",
            traj_len=1000, val_split=True, return_robot_obs=True, return_unprocessed_actions=True)]
        recorded_traj = traj_list[1]
        goal = recorded_traj['instruction']

    else:
        recorded_traj = None

        # goal = "move the slider left"
        goal = "turn on the green light"
        # goal = "turn on the blue light"
        # goal = "turn on the red light"
        # goal = "move the slider right"
        # goal = "stack the blue block on the green block"
        # goal = "unstack the blue block"
        # goal = "open the drawer"

    print("start episode")

    lang_rollout(model, env, goal, hist_len, ep_len=500, gt_traj=recorded_traj)


if __name__ == "__main__":
    # main()
    main(None)
