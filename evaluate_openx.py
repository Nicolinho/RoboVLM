import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from accelerate import Accelerator
from os.path import join as pjoin
from PIL import Image



def tfa_action_to_bridge_action(tfa_action):
  return np.concatenate((tfa_action['world_vector'], tfa_action['rotation_delta'], tfa_action['gripper_closedness_action']))

def visualize_actions(predicted_actions, gt_actions, resized_images, goal, title="", eval_result_dir='/tmp'):
    print("Generate plot for with title: ", title)
    # predicted_actions_bridge_format = np.array(list(map(tfa_action_to_bridge_action, predicted_actions)))
    predicted_actions_bridge_format = np.stack(predicted_actions)
    # predicted_actions_bridge_format = predicted_actions[None,:]
    action_order = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'grasp_continuous']

    gt_actions = np.stack(gt_actions)
    # gt_actions = gt_actions[None,:]



    plt.rcParams.update({'font.size': 12})

    # stacked = tf.concat(tf.unstack(resized_images[::3], axis=0), 1)
    stacked_img = resized_images[:gt_actions.shape[0]][::gt_actions.shape[0]//7][:len(action_order)]
    # stacked_img = resized_images[[0,5,20,30,40,50, 100]]
    # stacked_img = resized_images[:gt_actions.shape[0]][::3]

    figure_layout = [
        # ['image'] * len(stacked_img),
        [f"image{i}" for i in range(len(stacked_img))],
        action_order
    ]

    # fig, axs = plt.subplots(1, len(action_name_to_values_over_time))
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([45, 10])

    for action_dim, action_name in enumerate(action_order):
      axs[action_name].plot(predicted_actions_bridge_format[:, action_dim], label='predicted action')
      axs[action_name].plot(gt_actions[:, action_dim], label='ground truth')

      axs[action_name].set_title(action_name)
      axs[action_name].set_ylim(-1.1, 1.1)
      axs[action_name].set_xlabel('Time in one episode')

    for i in range(stacked_img.shape[0]):
        axs[f"image{i}"].imshow(stacked_img[i])
        # axs['image'].imshow(resized_images[action_dim])

    # axs['image'].set_xlabel('Time in one episode (subsampled)')
    # axs['image'].set_title(f'{title=}')

    axs['image3'].set_title(goal)

    plt.legend()

    # plt.savefig('/home/dorka/experiments/eval_openx_traj/eval_traj4.jpg')
    # plt.savefig(f"/home/dorka/experiments/eval_openx_traj/eval_{title}.jpg")
    plt.savefig(pjoin(eval_result_dir, f"eval_{title}.jpg"))

    # plt.imshow(stacked_img[0]-stacked_img[-1])
    # plt.savefig('/home/dorka/experiments/eval_openx_traj/img_diff.jpg')


    plt.pause(1)



from pathlib import Path
import os
# rtx relevant import
# from tf_agents.trajectories import time_step as ts
# from tf_agents.policies import py_tf_eager_policy


import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
import torch
import hydra
import cv2
from datetime import datetime

# from hulc2.evaluation.utils import imshow_tensor
# # from hulc2.models.hulc2 import Hulc2
# # from hulc2.utils.utils import format_sftp_path, get_checkpoints_for_epochs
# from hulc2.wrappers.panda_rtx_wrapper import PandaRTXWrapper
# from calvin_env.utils.utils import angle_between_angles
# from robot_io.utils.utils import quat_to_euler







# def load_model(cfg):
#     tfa_policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
#         model_path=cfg.checkpoint_dir,
#         load_specs_from_pbtxt=True,
#         use_tf_function=True)
#     print("finish loading policy")
#
#     # test the model with one frame of observation
#     observation = {
#         'image':
#             np.zeros(shape=(256, 320, 3), dtype=np.uint8),
#         'natural_language_instruction':
#             np.zeros(shape=(), dtype=str),
#         'natural_language_embedding':
#             np.zeros(shape=(512), dtype=np.float32),
#         # 'gripper_closed':
#         #     np.zeros(shape=(1), dtype=np.float32),
#         # 'height_to_bottom':
#         #     np.zeros(shape=(1), dtype=np.float32),
#         # 'base_pose_tool_reached':
#         #     np.zeros(shape=(7), dtype=np.float32),
#         # 'workspace_bounds':
#         #     np.zeros(shape=(3, 3), dtype=np.float32),
#         # 'orientation_box': np.zeros(shape=(2, 3), dtype=np.float32), 'orientation_start':
#         #     np.zeros(shape=(4), dtype=np.float32),
#         # 'src_rotation':
#         #     np.zeros(shape=(4), dtype=np.float32),
#         # 'robot_orientation_positions_box':
#         #     np.zeros(shape=(3, 3), dtype=np.float32),
#         # 'vector_to_go':
#         #     np.zeros(shape=(3), dtype=np.float32),
#         # 'rotation_delta_to_go':
#         #     np.zeros(shape=(3), dtype=np.float32),
#         # 'gripper_closedness_commanded':
#         #     np.zeros(shape=(1), dtype=np.float32),
#     }
#
#     tfa_time_step = ts.transition(observation, reward=np.zeros(()))
#
#     policy_state = tfa_policy.get_initial_state(batch_size=1)
#
#     action = tfa_policy.action(tfa_time_step, policy_state)
#     print(f"RTX model initialized. Action space: {action.action}")
#     return tfa_policy


def lang_rollout(model, env, goal, hist_len, ep_len=500):
    print("Type your instruction which the robot will try to follow")
    # while 1:
    #     lang_input = [input("What should I do? \n")]
    #     goal = lang_input[0]
    #     print("sleeping 5 seconds...)")
    #     time.sleep(6)
    #     rollout(env, model, goal, embed_model)
    print(goal)
    rollout(env, model, goal, hist_len, ep_len=ep_len)


def model_step(model, obs, policy_state, goal, goal_embed):
    image = tf.cast(tf.image.resize_with_pad(obs['rgb_static'], target_width=320, target_height=256), np.uint8)
    print(image.shape)
    observation = {
        'image': image.numpy(),
        'natural_language_instruction': goal,
        'natural_language_embedding': goal_embed,
        # 'gripper_closed':
        # np.zeros(shape=(1), dtype=np.float32),
        # 'height_to_bottom':
        #     np.zeros(shape=(1), dtype=np.float32),
        # 'base_pose_tool_reached':
        #     np.zeros(shape=(7), dtype=np.float32),
        # 'workspace_bounds':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'orientation_box':
        #     np.zeros(shape=(2, 3), dtype=np.float32),
        # 'orientation_start':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'src_rotation':
        #     np.zeros(shape=(4), dtype=np.float32),
        # 'robot_orientation_positions_box':
        #     np.zeros(shape=(3, 3), dtype=np.float32),
        # 'vector_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'rotation_delta_to_go':
        #     np.zeros(shape=(3), dtype=np.float32),
        # 'gripper_closedness_commanded':
        #     np.zeros(shape=(1), dtype=np.float32),
    }
    tfa_time_step = ts.transition(observation, reward=np.zeros(()))
    policy_step = model.action(tfa_time_step, policy_state)

    return policy_step


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


def tfa_action_to_bridge_action(tfa_action):
    return np.concatenate((tfa_action['world_vector'], tfa_action['rotation_delta'], tfa_action['gripper_closedness_action']))


def obs_dict_to_np(robot_obs):
    tcp_pos = robot_obs["tcp_pos"]
    tcp_orn = quat_to_euler(robot_obs["tcp_orn"])
    gripper_width = robot_obs["gripper_opening_width"]
    gripper_action = 1 if gripper_width > 0.06 else -1

    return np.concatenate([tcp_pos, tcp_orn, [gripper_action]])


def rollout(env, model, goal, hist_len, ep_len=5000):
    # env.reset()
    # obs = env.get_obs()
    obs = {'rgb_static': np.random.randint(0, 255, size = (200,200,3), dtype=np.uint8)}

    model.reset_history(instruction=goal, max_len=hist_len)
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # dd/mm/YY H:M:S
    goal_str = "_".join(goal.split()) + "_imgs"
    dt_string = now.strftime("%Y_%m_%d_at_%H_%M_%S")
    folder = Path("/tmp") / goal_str / dt_string
    os.makedirs(folder, exist_ok=True)

    for step in range(ep_len):

        action = model.select_action(obs['rgb_static'])
        print(action)
        action = _unscale_action(np.array(action))

        # curr_pose = obs_dict_to_np(obs["robot_state"])
        # rel_act = to_relative_action(action, curr_pose) #TODO can this be done in torch?
        # rel_act_torch = torch.tensor(rel_act)

        ## rel_act_torch = torch.tensor(action)

        # obs, _, _, _ = env.step(rel_act_torch)
        #
        # cv2.imshow("rgb_static", obs["rgb_static"][:, :, ::-1])
        # save_path = folder / f"{step:03}.png"
        # cv2.imwrite(save_path.as_posix(), obs["rgb_static"][:, :, ::-1])

        k = cv2.waitKey(200)

        # k = imshow_tensor("rgb_static", obs["rgb_static"], wait=1, resize=True, text=goal)
        # press ESC to stop rollout and return
        if k == 27:
            return


def evaluate_on_fixed_trajectory(model, hist_len, traj, name="", eval_result_dir="/tmp", from_data_loader=False):

    if from_data_loader:
        goal = traj['instruction'][0]
        traj['images'] = traj['images'][0]
        traj['actions_unprocessed'] = traj['actions_unprocessed'][0].cpu().numpy()
    else:
        goal = traj['instruction']

    # obs = {'rgb_static': np.random.randint(0, 255, size = (200,200,3), dtype=np.uint8)}
    # obs = {'rgb_static': traj['images'][0]}

    model.reset_history(instruction=goal, max_len=hist_len)
    # datetime object containing current date and time
    now = datetime.now()
    print("now =", now)

    # # dd/mm/YY H:M:S
    # goal_str = "_".join(goal.split()) + "_imgs"
    # dt_string = now.strftime("%Y_%m_%d_at_%H_%M_%S")
    # folder = Path("/tmp") / goal_str / dt_string
    # os.makedirs(folder, exist_ok=True)

    pred_actions = []
    gt_actions = [] # traj['actions_unprocessed']

    num_eval_actions = 200

    for step in range(min(len(traj['images']), num_eval_actions)):
        if from_data_loader:
            obs = {'rgb_static': traj['images'][step][None, :]}
        else:
            obs = {'rgb_static': Image.fromarray(traj['images'][step])}

        action = model.select_action(obs['rgb_static'])
        # print(action)

        # action = _unscale_action(np.array(action))
        # gt_action = _unscale_action(np.array(traj['actions_unprocessed'][step]))

        action = np.array(action)
        gt_action = np.array(traj['actions_unprocessed'][step])

        # curr_pose = obs_dict_to_np(obs["robot_state"]) # TODO what is the input action format to the visualizer
        # rel_act = to_relative_action(action, curr_pose)
        # rel_act_torch = torch.tensor(rel_act)

        pred_actions.append(action)
        gt_actions.append(gt_action)
        if action is None:
            print("Invalid action, end evaluation for this trajectory | ", name)
            break

        # visualize_actions(action, gt_action, traj['images'][step])

        # rel_act_torch = torch.tensor(action)

        # obs, _, _, _ = env.step(rel_act_torch)
        #


        # # cv2.imshow("rgb_static", obs["rgb_static"][:, :, ::-1])
        # # save_path = folder / f"{step:03}.png"
        # # cv2.imwrite(save_path.as_posix(), obs["rgb_static"][:, :, ::-1])
        #
        # k = cv2.waitKey(200)
        #
        # # k = imshow_tensor("rgb_static", obs["rgb_static"], wait=1, resize=True, text=goal)
        # # press ESC to stop rollout and return
        # if k == 27:
        #     return

    if from_data_loader:
        traj['images'] = traj['images'].cpu().numpy()
    visualize_actions(pred_actions, gt_actions, traj['images'], goal, f"traj_{name}", eval_result_dir)

# @hydra.main(config_path="../../conf", config_name="inference_real_rtx")
def main(cfg):
    from palme_model_openx import Palme

    # load robot
    # robot = hydra.utils.instantiate(cfg.robot)
    # env = hydra.utils.instantiate(cfg.env, robot=robot)
    # env = PandaRTXWrapper(env, relative_action=True)

    robot, env = None, None

    checkpoint_dir_path = "/home/dorka/chkpts/run_openx_test/checkpoint-865"
    checkpoint_dir_path = "/home/dorka/chkpts/taco_extradata_vit/checkpoint-435"
    # checkpoint_dir_path = "/home/dorka/chkpts/taco_extradata_clip/checkpoint-435"

    acces_token = "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"
    llama_checkpoint = "meta-llama/Llama-2-7b-hf"

    checkpoint_image_model = "google/vit-base-patch16-224-in21k"
    # checkpoint_image_model = "openai/clip-vit-base-patch32"
    # checkpoint_image_model = "google/vit-base-patch32-384"

    device_index = Accelerator().process_index
    device_map = {"": device_index}


    # for inference
    hist_len = 5

    # model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
    #               config=None, output_dir = None,
    #               load_in_8bit=True,
    #               lora_lm=True, lora_vision=False, freeze_vision=True,
    #               device_map=device_map,
    #               torch_dtype = torch.bfloat16,
    #               # torch_dtype = torch.float16,
    #               )


    model = Palme(llama_checkpoint=llama_checkpoint, acces_token=acces_token, image_model_name=checkpoint_image_model,
                  # config=None,
                  load_in_8bit=False,
                  # load_in_8bit=True,
                  lora_lm=False,
                  # lora_lm=True,
                  lora_vision=False, freeze_vision=True,
                  device_map=device_map,
                  torch_dtype = torch.bfloat16,
                  )

    ## model.load("/home/dorka/projects/23/llama-openx/palme/palme/taco_alldata/checkpoint-880/pytorch_model.bin")
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

    model.lm.load_state_dict(torch.load(
        pjoin(checkpoint_dir_path, 'dequant', 'lm_model.bin'), map_location="cpu"))

    if 'openai/clip' in  checkpoint_image_model:
        model.proj_layer.load_state_dict(torch.load(
            pjoin(checkpoint_dir_path, 'dequant', 'img_proj_layer_model.bin'), map_location="cpu"))
    elif 'google/vit' in  checkpoint_image_model:
        model.img_embed_model.classifier.load_state_dict(torch.load(
            pjoin(checkpoint_dir_path, 'dequant', 'img_embed_model_classifier_model.bin'), map_location="cpu"))

    model.do_torch_compile()
    # save_pretrained to be able to load it faster

    goal = "move the slider left"
    # goal = "turn on the green light"
    # goal = "turn on the blue light"
    # goal = "turn on the red light"
    # goal = "move the slider right"
    # goal = "stack the blue block on the green block"
    # goal = "unstack the blue block"
    # goal = "open the drawer"
    # lang_rollout(model, env, goal, hist_len, ep_len=500)


    from dataset_tools_openx import generator_taco_extra_data

    traj_list = [traj for traj in generator_taco_extra_data(
        # data_path="/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
        data_path="/export/home/huang/taco_extra_processed_15hz_resize",
        traj_len=1000, val_split=True, return_robot_obs=True, return_unprocessed_actions=True)]

    eval_result_dir = pjoin(checkpoint_dir_path, 'eval_results_no_unscale')
    os.makedirs(eval_result_dir, exist_ok=True)

    for i, traj in enumerate(traj_list):
        evaluate_on_fixed_trajectory(model, hist_len, traj, str(i), eval_result_dir)

if __name__ == "__main__":
    # main()
    main(None)
