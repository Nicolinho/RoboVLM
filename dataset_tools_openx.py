import tensorflow as tf
from datasets import load_dataset, Dataset
import os, json

import tqdm

from typing import Any, Dict, Union, NamedTuple

import numpy as np
import reverb
from rlds import transformations
import tensorflow_datasets as tfds
import tree

import abc
import dataclasses
from typing import Dict, Optional

from rlds import rlds_types
from PIL import Image


NUM_BINS = 100
ACTION_BINS = np.linspace(-1, 1, NUM_BINS)
ACTION_BINS_GRIPPER = np.linspace(0, 1, NUM_BINS)



def text_to_action(text, gripper_range_2):
    """decodes a string into the action format of openx and returns the action_type and if required its parameters"""
    try:
        disc_actions = text.split(" ")[:7]
        if gripper_range_2:
            actions = [ACTION_BINS[int(da)] for da in disc_actions[:6]] + [ACTION_BINS[int(disc_actions[6])]]
        else:
            actions = [ACTION_BINS[int(da)] for da in disc_actions[:6]] + [ACTION_BINS_GRIPPER[int(disc_actions[6])]]
        return actions
    except Exception as e:
        print("Error in text to action for action: ", text, " | ERROR: ", e)
        return None





def get_link_manual_download(dataset_name):
    DATASETS = [
        'fractal20220817_data',
        'kuka',
        'bridge',
        'taco_play',
        'jaco_play',
        'berkeley_cable_routing',
        'roboturk',
        'nyu_door_opening_surprising_effectiveness',
        'viola',
        'berkeley_autolab_ur5',
        'toto',
        'language_table',
        'columbia_cairlab_pusht_real',
        'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
        'nyu_rot_dataset_converted_externally_to_rlds',
        'stanford_hydra_dataset_converted_externally_to_rlds',
        'austin_buds_dataset_converted_externally_to_rlds',
        'nyu_franka_play_dataset_converted_externally_to_rlds',
        'maniskill_dataset_converted_externally_to_rlds',
        'cmu_franka_exploration_dataset_converted_externally_to_rlds',
        'ucsd_kitchen_dataset_converted_externally_to_rlds',
        'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
        'austin_sailor_dataset_converted_externally_to_rlds',
        'austin_sirius_dataset_converted_externally_to_rlds',
        'bc_z',
        'usc_cloth_sim_converted_externally_to_rlds',
        'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
        'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
        'utokyo_saytap_converted_externally_to_rlds',
        'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
        'utokyo_xarm_bimanual_converted_externally_to_rlds',
        'robo_net',
        'berkeley_mvp_converted_externally_to_rlds',
        'berkeley_rpt_converted_externally_to_rlds',
        'kaist_nonprehensile_converted_externally_to_rlds',
        'stanford_mask_vit_converted_externally_to_rlds',
        'tokyo_u_lsmo_converted_externally_to_rlds',
        'dlr_sara_pour_converted_externally_to_rlds',
        'dlr_sara_grid_clamp_converted_externally_to_rlds',
        'dlr_edan_shared_control_converted_externally_to_rlds',
        'asu_table_top_converted_externally_to_rlds',
        'stanford_robocook_converted_externally_to_rlds',
        'eth_agent_affordances',
        'imperialcollege_sawyer_wrist_cam',
        'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
        'uiuc_d3field',
        'utaustin_mutex',
        'berkeley_fanuc_manipulation',
        'cmu_play_fusion',
        'cmu_stretch',
        'berkeley_gnm_recon',
        'berkeley_gnm_cory_hall',
        'berkeley_gnm_sac_son'
    ]

    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    else:
        version = '0.1.0'
    return f'gs://gresearch/robotics/{dataset_name}/{version}'


# @title Transformation definitions

def _features_to_tensor_spec(
    feature: tfds.features.FeatureConnector
) -> tf.TensorSpec:
  """Converts a tfds Feature into a TensorSpec."""

  def _get_feature_spec(nested_feature: tfds.features.FeatureConnector):
    if isinstance(nested_feature, tf.DType):
      return tf.TensorSpec(shape=(), dtype=nested_feature)
    else:
      return nested_feature.get_tensor_spec()

  # FeaturesDict can sometimes be a plain dictionary, so we use tf.nest to
  # make sure we deal with the nested structure.
  return tf.nest.map_structure(_get_feature_spec, feature)


def _encoded_feature(feature: Optional[tfds.features.FeatureConnector],
                     image_encoding: Optional[str],
                     tensor_encoding: Optional[tfds.features.Encoding]):
  """Adds encoding to Images and/or Tensors."""
  def _apply_encoding(feature: tfds.features.FeatureConnector,
                      image_encoding: Optional[str],
                      tensor_encoding: Optional[tfds.features.Encoding]):
    if image_encoding and isinstance(feature, tfds.features.Image):
      return tfds.features.Image(
          shape=feature.shape,
          dtype=feature.dtype,
          use_colormap=feature.use_colormap,
          encoding_format=image_encoding)
    if tensor_encoding and isinstance(
        feature, tfds.features.Tensor) and feature.dtype != tf.string:
      return tfds.features.Tensor(
          shape=feature.shape, dtype=feature.dtype, encoding=tensor_encoding)
    return feature

  if not feature:
    return None
  return tf.nest.map_structure(
      lambda x: _apply_encoding(x, image_encoding, tensor_encoding), feature)


@dataclasses.dataclass
class RLDSSpec(metaclass=abc.ABCMeta):
  """Specification of an RLDS Dataset.

  It is used to hold a spec that can be converted into a TFDS DatasetInfo or
  a `tf.data.Dataset` spec.
  """
  observation_info: Optional[tfds.features.FeatureConnector] = None
  action_info: Optional[tfds.features.FeatureConnector] = None
  reward_info: Optional[tfds.features.FeatureConnector] = None
  discount_info: Optional[tfds.features.FeatureConnector] = None
  step_metadata_info: Optional[tfds.features.FeaturesDict] = None
  episode_metadata_info: Optional[tfds.features.FeaturesDict] = None

  def step_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
    """Obtains the TensorSpec of an RLDS step."""
    step = {}
    if self.observation_info:
      step[rlds_types.OBSERVATION] = _features_to_tensor_spec(
          self.observation_info)
    if self.action_info:
      step[rlds_types.ACTION] = _features_to_tensor_spec(
          self.action_info)
    if self.discount_info:
      step[rlds_types.DISCOUNT] = _features_to_tensor_spec(
          self.discount_info)
    if self.reward_info:
      step[rlds_types.REWARD] = _features_to_tensor_spec(
          self.reward_info)
    if self.step_metadata_info:
      for k, v in self.step_metadata_info.items():
        step[k] = _features_to_tensor_spec(v)

    step[rlds_types.IS_FIRST] = tf.TensorSpec(shape=(), dtype=bool)
    step[rlds_types.IS_LAST] = tf.TensorSpec(shape=(), dtype=bool)
    step[rlds_types.IS_TERMINAL] = tf.TensorSpec(shape=(), dtype=bool)
    return step

  def episode_tensor_spec(self) -> Dict[str, tf.TensorSpec]:
    """Obtains the TensorSpec of an RLDS step."""
    episode = {}
    episode[rlds_types.STEPS] = tf.data.DatasetSpec(
        element_spec=self.step_tensor_spec())
    if self.episode_metadata_info:
      for k, v in self.episode_metadata_info.items():
        episode[k] = _features_to_tensor_spec(v)
    return episode

  def to_dataset_config(
      self,
      name: str,
      image_encoding: Optional[str] = None,
      tensor_encoding: Optional[tfds.features.Encoding] = None,
      citation: Optional[str] = None,
      homepage: Optional[str] = None,
      description: Optional[str] = None,
      overall_description: Optional[str] = None,
  ) -> tfds.rlds.rlds_base.DatasetConfig:
    """Obtains the DatasetConfig for TFDS from the Spec."""
    return tfds.rlds.rlds_base.DatasetConfig(
        name=name,
        description=description,
        overall_description=overall_description,
        homepage=homepage,
        citation=citation,
        observation_info=_encoded_feature(self.observation_info, image_encoding,
                                          tensor_encoding),
        action_info=_encoded_feature(self.action_info, image_encoding,
                                     tensor_encoding),
        reward_info=_encoded_feature(self.reward_info, image_encoding,
                                     tensor_encoding),
        discount_info=_encoded_feature(self.discount_info, image_encoding,
                                       tensor_encoding),
        step_metadata_info=_encoded_feature(self.step_metadata_info,
                                            image_encoding, tensor_encoding),
        episode_metadata_info=_encoded_feature(self.episode_metadata_info,
                                               image_encoding, tensor_encoding))

  def to_features_dict(self):
    """Returns a TFDS FeaturesDict representing the dataset config."""
    step_config = {
        rlds_types.IS_FIRST: tf.bool,
        rlds_types.IS_LAST: tf.bool,
        rlds_types.IS_TERMINAL: tf.bool,
    }

    if self.observation_info:
      step_config[rlds_types.OBSERVATION] = self.observation_info
    if self.action_info:
      step_config[rlds_types.ACTION] = self.action_info
    if self.discount_info:
      step_config[rlds_types.DISCOUNT] = self.discount_info
    if self.reward_info:
      step_config[rlds_types.REWARD] = self.reward_info

    if self.step_metadata_info:
      for k, v in self.step_metadata_info.items():
        step_config[k] = v

    if self.episode_metadata_info:
      return tfds.features.FeaturesDict({
          rlds_types.STEPS: tfds.features.Dataset(step_config),
          **self.episode_metadata_info,
      })
    else:
      return tfds.features.FeaturesDict({
          rlds_types.STEPS: tfds.features.Dataset(step_config),
      })

RLDS_SPEC = RLDSSpec
TENSOR_SPEC = Union[tf.TensorSpec, dict[str, tf.TensorSpec]]


@dataclasses.dataclass
class TrajectoryTransform(metaclass=abc.ABCMeta):
  """Specification the TrajectoryTransform applied to a dataset of episodes.

  A TrajectoryTransform is a set of rules transforming a dataset
  of RLDS episodes to a dataset of trajectories.
  This involves three distinct stages:
  - An optional `episode_to_steps_map_fn(episode)` is called at the episode
    level, and can be used to select or modify steps.
    - Augmentation: an `episode_key` could be propagated to `steps` for
      debugging.
    - Selection: Particular steps can be selected.
    - Stripping: Features can be removed from steps. Prefer using `step_map_fn`.
  - An optional `step_map_fn` is called at the flattened steps dataset for each
    step, and can be used to featurize a step, e.g. add/remove features, or
    augument images
  - A `pattern` leverages DM patterns to set a rule of slicing an episode to a
    dataset of overlapping trajectories.

  Importantly, each TrajectoryTransform must define a `expected_tensor_spec`
  which specifies a nested TensorSpec of the resulting dataset. This is what
  this TrajectoryTransform will produce, and can be used as an interface with
  a neural network.
  """
  episode_dataset_spec: RLDS_SPEC
  episode_to_steps_fn_dataset_spec: RLDS_SPEC
  steps_dataset_spec: Any
  pattern: reverb.structured_writer.Pattern
  episode_to_steps_map_fn: Any
  expected_tensor_spec: TENSOR_SPEC
  step_map_fn: Optional[Any] = None

  def get_for_cached_trajectory_transform(self):
    """Creates a copy of this traj transform to use with caching.

    The returned TrajectoryTransfrom copy will be initialized with the default
    version of the `episode_to_steps_map_fn`, because the effect of that
    function has already been materialized in the cached copy of the dataset.
    Returns:
      trajectory_transform: A copy of the TrajectoryTransform with overridden
        `episode_to_steps_map_fn`.
    """
    traj_copy = dataclasses.replace(self)
    traj_copy.episode_dataset_spec = traj_copy.episode_to_steps_fn_dataset_spec
    traj_copy.episode_to_steps_map_fn = lambda e: e[rlds_types.STEPS]
    return traj_copy

  def transform_episodic_rlds_dataset(self, episodes_dataset: tf.data.Dataset):
    """Applies this TrajectoryTransform to the dataset of episodes."""

    # Convert the dataset of episodes to the dataset of steps.
    steps_dataset = episodes_dataset.map(
        self.episode_to_steps_map_fn, num_parallel_calls=tf.data.AUTOTUNE
    ).flat_map(lambda x: x)

    return self._create_pattern_dataset(steps_dataset)

  def transform_steps_rlds_dataset(
      self, steps_dataset: tf.data.Dataset
  ) -> tf.data.Dataset:
    """Applies this TrajectoryTransform to the dataset of episode steps."""

    return self._create_pattern_dataset(steps_dataset)

  def create_test_dataset(
      self,
  ) -> tf.data.Dataset:
    """Creates a test dataset of trajectories.

    It is guaranteed that the structure of this dataset will be the same as
    when flowing real data. Hence this is a useful construct for tests or
    initialization of JAX models.
    Returns:
      dataset: A test dataset made of zeros structurally identical to the
        target dataset of trajectories.
    """
    zeros = transformations.zeros_from_spec(self.expected_tensor_spec)

    return tf.data.Dataset.from_tensors(zeros)

  def _create_pattern_dataset(
      self, steps_dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Create PatternDataset from the `steps_dataset`."""
    config = create_structured_writer_config('temp', self.pattern)

    # Further transform each step if the `step_map_fn` is provided.
    if self.step_map_fn:
      steps_dataset = steps_dataset.map(self.step_map_fn)
    pattern_dataset = reverb.PatternDataset(
        input_dataset=steps_dataset,
        configs=[config],
        respect_episode_boundaries=True,
        is_end_of_episode=lambda x: x[rlds_types.IS_LAST])
    return pattern_dataset


class TrajectoryTransformBuilder(object):
  """Facilitates creation of the `TrajectoryTransform`."""

  def __init__(self,
               dataset_spec: RLDS_SPEC,
               episode_to_steps_map_fn=lambda e: e[rlds_types.STEPS],
               step_map_fn=None,
               pattern_fn=None,
               expected_tensor_spec=None):
    self._rds_dataset_spec = dataset_spec
    self._steps_spec = None
    self._episode_to_steps_map_fn = episode_to_steps_map_fn
    self._step_map_fn = step_map_fn
    self._pattern_fn = pattern_fn
    self._expected_tensor_spec = expected_tensor_spec

  def build(self,
            validate_expected_tensor_spec: bool = True) -> TrajectoryTransform:
    """Creates `TrajectoryTransform` from a `TrajectoryTransformBuilder`."""

    if validate_expected_tensor_spec and self._expected_tensor_spec is None:
      raise ValueError('`expected_tensor_spec` must be set.')

    episode_ds = zero_episode_dataset_from_spec(self._rds_dataset_spec)

    steps_ds = episode_ds.flat_map(self._episode_to_steps_map_fn)

    episode_to_steps_fn_dataset_spec = self._rds_dataset_spec

    if self._step_map_fn is not None:
      steps_ds = steps_ds.map(self._step_map_fn)

    zeros_spec = transformations.zeros_from_spec(steps_ds.element_spec)  # pytype: disable=wrong-arg-types

    ref_step = reverb.structured_writer.create_reference_step(zeros_spec)

    pattern = self._pattern_fn(ref_step)

    steps_ds_spec = steps_ds.element_spec

    target_tensor_structure = create_reverb_table_signature(
        'temp_table', steps_ds_spec, pattern)

    if (validate_expected_tensor_spec and
        self._expected_tensor_spec != target_tensor_structure):
      raise RuntimeError(
          'The tensor spec of the TrajectoryTransform doesn\'t '
          'match the expected spec.\n'
          'Expected:\n%s\nActual:\n%s\n' %
          (str(self._expected_tensor_spec).replace('TensorSpec',
                                                   'tf.TensorSpec'),
           str(target_tensor_structure).replace('TensorSpec', 'tf.TensorSpec')))

    return TrajectoryTransform(
        episode_dataset_spec=self._rds_dataset_spec,
        episode_to_steps_fn_dataset_spec=episode_to_steps_fn_dataset_spec,
        steps_dataset_spec=steps_ds_spec,
        pattern=pattern,
        episode_to_steps_map_fn=self._episode_to_steps_map_fn,
        step_map_fn=self._step_map_fn,
        expected_tensor_spec=target_tensor_structure)

def zero_episode_dataset_from_spec(rlds_spec: RLDS_SPEC):
  """Creates a zero valued dataset of episodes for the given RLDS Spec."""

  def add_steps(episode, step_spec):
    episode[rlds_types.STEPS] = transformations.zero_dataset_like(
        tf.data.DatasetSpec(step_spec))
    if 'fake' in episode:
      del episode['fake']
    return episode

  episode_without_steps_spec = {
      k: v
      for k, v in rlds_spec.episode_tensor_spec().items()
      if k != rlds_types.STEPS
  }

  if episode_without_steps_spec:
    episodes_dataset = transformations.zero_dataset_like(
        tf.data.DatasetSpec(episode_without_steps_spec))
  else:
    episodes_dataset = tf.data.Dataset.from_tensors({'fake': ''})

  episodes_dataset_with_steps = episodes_dataset.map(
      lambda episode: add_steps(episode, rlds_spec.step_tensor_spec()))
  return episodes_dataset_with_steps


def create_reverb_table_signature(table_name: str, steps_dataset_spec,
                                  pattern: reverb.structured_writer.Pattern) -> reverb.reverb_types.SpecNest:
  config = create_structured_writer_config(table_name, pattern)
  reverb_table_spec = reverb.structured_writer.infer_signature(
      [config], steps_dataset_spec)
  return reverb_table_spec


def create_structured_writer_config(table_name: str,
                                    pattern: reverb.structured_writer.Pattern) -> Any:
  config = reverb.structured_writer.create_config(
      pattern=pattern, table=table_name, conditions=[])
  return config

def n_step_pattern_builder(n: int) -> Any:
  """Creates trajectory of length `n` from all fields of a `ref_step`."""

  def transform_fn(ref_step):
    traj = {}
    for key in ref_step:
      if isinstance(ref_step[key], dict):
        transformed_entry = tree.map_structure(lambda ref_node: ref_node[-n:],
                                               ref_step[key])
        traj[key] = transformed_entry
      else:
        traj[key] = ref_step[key][-n:]

    return traj

  return transform_fn



def data_generator_openx(dataset, limit=1000000000):
    for i, episode in enumerate(dataset):
        if i >= limit:
            break
        yield process_episode(episode)

def process_episode(episode):
    print(episode)
    display_key = 'image'
    # images = [step['observation'][display_key] for step in episode['steps']]
    actions = [tf.concat([  step['action']['world_vector'],
                            step['action']['rotation_delta'],
                            step['action']['gripper_closedness_action'],
                          ], axis=-1) for step in episode['steps']]
    return {"images": images, "actions": actions}

def process_action(actions, gripper_range_2):
    if gripper_range_2: # whether the gripper action is in the range [-1,1] or [0,1]
        disc_actions = np.concatenate((np.clip(np.digitize(actions[:-1], ACTION_BINS, right=True),  a_min=0, a_max=len(ACTION_BINS) -1),
                                       np.clip(np.array([np.digitize(actions[-1], ACTION_BINS, right=True)]),  a_min=0, a_max=len(ACTION_BINS) -1)))
    else:
        disc_actions = np.concatenate((np.clip(np.digitize(actions[:-1], ACTION_BINS, right=True),  a_min=0, a_max=len(ACTION_BINS) -1),
                                       np.clip(np.array([np.digitize(actions[-1], ACTION_BINS_GRIPPER, right=True)]),  a_min=0, a_max=len(ACTION_BINS_GRIPPER) -1)))
    action_string = " ".join([str(a) for a in disc_actions])
    return action_string


def generator_fun_openx(builder_dir, traj_len=3, limit=100000000):
    b = tfds.builder_from_directory(builder_dir=builder_dir)
    # ds = b.as_dataset(split='train') #train[:10]
    ds = b.as_dataset(split='train[:10]') #train[:10]
    mt_opt_rlds_spec = RLDSSpec(
        observation_info=b.info.features['steps']['observation'],
        action_info=b.info.features['steps']['action'],
    )

    # The following will create a trajectories of length 3.
    trajectory_length = traj_len
    trajectory_transform = TrajectoryTransformBuilder(mt_opt_rlds_spec,
                                                      pattern_fn=n_step_pattern_builder(trajectory_length)).build(
                                                      validate_expected_tensor_spec=False)
    trajectory_dataset = trajectory_transform.transform_episodic_rlds_dataset(ds)

    tds_np = trajectory_dataset.as_numpy_iterator()

    for i, traj in enumerate(tds_np):
        if i >= limit:
            break
        # yield episode
        # print(episode['observation']['image'])
        # break
        # yield {'image': traj['observation']['image']}
        instruction = traj['observation']['natural_language_instruction'][0].decode("utf-8")
        images = traj['observation']['image']
        actions = [np.concatenate((x,y,z)) for x,y,z in zip(traj['action']['world_vector'], traj['action']['rotation_delta'], traj['action']['gripper_closedness_action'])]
        actions_strings = [process_action(a) for a in actions]
        yield {'images': images, 'instruction': instruction, 'actions': actions_strings}
        # yield traj
        # yield process_episode(episode)
        #
def generator_fun_openx_nooverlap(builder_dir, taco_extra_data_dir=None, traj_len=3, img_resize_dim=None, shards=None, limit=100000000):
    if builder_dir is not None:
        b = tfds.builder_from_directory(builder_dir=builder_dir)
        if shards is None:
            ds = b.as_dataset(split='train') #train[:10]
        else:
            ds = b.as_dataset(split="train" + shards[0]) #train[:10]
            # ds = b.as_dataset(split=f"train[:10]") #train[:10]
        ## ds = ds.as_numpy_iterator()

        ds_size = 0

        for d in iter(ds):
            if ds_size > limit:
                break
            # episode = [obs for obs in d['steps']['observations']]
            if img_resize_dim is None:
                images = np.stack([step['observation']['image'].numpy() for step in d['steps']])
            else:
                # images = np.stack([tf.cast(tf.image.resize(step['observation']['image'], img_resize_dim), tf.uint8).numpy() for step in d['steps']])
                # for taco
                images = np.stack([tf.cast(tf.image.resize(step['observation']['rgb_static'], img_resize_dim), tf.uint8).numpy() for step in d['steps']])

            # actions = [tf.concat([
            #     step['action']['world_vector'],
            #     step['action']['rotation_delta'],
            #     step['action']['gripper_closedness_action'],
            #                       ], axis=-1).numpy()
            #            for step in d['steps']]

            #for taco #TODO
            actions = [taco_data_scale_action_openx(step['action']['actions'].numpy()) for step in  d['steps']]
            actions_strings = [process_action(a, gripper_range_2=True) for a in actions]

            # actions_strings = [process_action(a) for a in actions]

            instruction = next(iter(d['steps']))['observation']['natural_language_instruction'].numpy().decode("utf-8")

            for i in range(0, len(images), traj_len):
                ds_size += 1
                yield {'images': images[i:(i+traj_len)], 'actions': actions_strings[i:(i+traj_len)], 'instruction': instruction}

            if i < len(images) - traj_len:
                ds_size += 1
                yield {'images': images[-traj_len:], 'actions': actions_strings[-traj_len:], 'instruction': instruction}

    if taco_extra_data_dir is not None and (shards is None or shards[0].startswith('[0')):
        # this gen does not use sharding, so in case of sharding only process the data in the first process
        for traj in generator_taco_extra_data(taco_extra_data_dir,traj_len, img_resize_dim):
            yield traj

def taco_data_scale_action_openx(action):
    # Actions is SymbolicTensor, shape (N,)
    # Rotation Delta
    rd_lows = tf.constant([-3.2, -0.8, -1.8])
    rd_highs = tf.constant([3.2, 0.2, 2.5])
    action[3:6] = _rescale_actions_by_bounds(action[3:6], rd_lows, rd_highs, safety_margin=0.01)

    # World Vector
    wv_lows = tf.constant([0.0, -0.5, 0.0])
    wv_highs = tf.constant([0.8, 0.7, 0.6])

    action[0:3] = _rescale_actions_by_bounds(action[0:3], wv_lows, wv_highs, safety_margin=0.01)

    return action

def _rescale_actions_by_bounds(actions, lows, highs, safety_margin=0.01):
  # Actions is SymbolicTensor, shape (N,)
  resc_actions = (actions - lows) / (highs - lows) * 2 - 1
  return np.clip(resc_actions, -1 + safety_margin, 1 - safety_margin)

def generator_taco_extra_data(data_path, traj_len=3, img_resize_dim=None, val_split=False, return_robot_obs=False,
                              return_unprocessed_actions=False):
    instruction_data = np.load(os.path.join(data_path, 'lang_paraphrase-MiniLM-L3-v2_singleTasks/auto_lang_ann.npy'),
                               allow_pickle=True).item()

    step_files = [fp for fp in os.listdir(data_path) if fp.endswith('.npz')]

    # val_episodes = [0]  # could make random selection
    # np.random.seed(7)
    # val_episodes = np.concatenate(
    #     (np.array([0]), np.random.choice(len(instruction_data['info']['indx']), 10)))  # could make random selection
    val_episodes = [  0, 175, 196, 502, 211, 348, 185, 398, 345, 366, 167]

    if val_split:

        iter_list = val_episodes
        print("Episde ids to eval on: ", iter_list)
        # iter_list = np.concatenate((np.array([0]), np.random.choice(100, 10))) # could make random selection
    else:
        iter_list = [i for i in  range(1, len(instruction_data['info']['indx'])) if i not in val_episodes]

    # skip first episode for evaluation
    for seq_nr in iter_list:
        instruction = instruction_data['language']['ann'][seq_nr]
        start, end = instruction_data['info']['indx'][seq_nr]
        images, image_sizes = [], []
        actions = []
        actions_unproc = []
        robot_obs = []
        for j in range(start, end+1):
            step = np.load(os.path.join(data_path, f"episode_" + str(j).zfill(7)) + ".npz")
            if img_resize_dim is not None:
                images.append(tf.cast(tf.image.resize(step['rgb_static'], img_resize_dim), tf.uint8).numpy())
            else:
                images.append(Image.fromarray(step['rgb_static']))
                image_sizes.append(images[-1].size)

            actions.append(taco_data_scale_action_openx(step['actions']))

            if return_robot_obs:
                robot_obs.append(step['robot_obs'])
                # array([0.30644304, 0.01565283, 0.49398055, 3.08235332, 0.01601486,
                #        0.05900232, 0.08059336, -0.01014491, -0.77117596, 0.06544485,
                #        -2.32554914, -0.01399427, 1.54077639, 0.76208691, 1.])

        images = np.stack(images)
        if return_robot_obs:
            robot_obs = np.stack(robot_obs)
        actions_strings = [process_action(a, gripper_range_2 = True) for a in actions]

        # print(actions)



        for i in range(0, len(images), traj_len):
            d = {'images': images[i:(i+traj_len)], 'actions': actions_strings[i:(i+traj_len)],
                 'instruction': instruction, 'image_sizes': image_sizes[i:(i+traj_len)]}
            if return_robot_obs:
                d['robot_obs'] = robot_obs[i:(i+traj_len)]
            if return_unprocessed_actions:
                d['actions_unprocessed'] = np.stack(actions[i:(i+traj_len)])
            if val_split:
                d['seq_nr'] = seq_nr
            yield d

        if i < len(images) - traj_len:
            d = {'images': images[-traj_len:], 'actions': actions_strings[-traj_len:],
                 'instruction': instruction, 'image_sizes': image_sizes[-traj_len:]}
            if return_robot_obs:
                d['robot_obs'] = robot_obs[-traj_len:]
            if return_unprocessed_actions:
                d['actions_unprocessed'] = np.stack(actions[-traj_len:])
            if val_split:
                d['seq_nr'] = seq_nr
            yield d

def num_proc_to_shard_string(num_proc):
    shard_size = 100 // num_proc
    return [f"[{i}%:{j}%]" for i, j in zip(range(0, 100, shard_size),
                                          range(shard_size, 100+shard_size, shard_size))]


if __name__=='__main__':
    ds_train = Dataset.from_generator(generator_taco_extra_data,
        gen_kwargs={
            'data_path': "/home/dorka/data/tensorflow_ds/taco_play/extra_data/taco_extra_processed_15hz_resize/",
            # 'limit': 1000000,
            # 'shards': num_proc_to_shard_string(10),
            "traj_len": 10}, #5
        num_proc=10, writer_batch_size=50)








