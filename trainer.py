import os
import re

import accelerate
import torch
import numpy as np
from transformers import Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
import wandb
import traceback

from dataset_tools_openx import text_to_action
from evaluate_openx import evaluate_on_fixed_trajectory

import traceback

from accelerate import Accelerator

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_model_param_count,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from transformers.utils import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    PushInProgress,
    can_return_loss,
    find_labels,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_datasets_available,
    is_in_notebook,
    is_ipex_available,
    is_peft_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_compile_available,
    is_torch_neuroncore_available,
    is_torch_tpu_available,
    logging,
    strtobool,
)

from transformers.trainer import (
    # is_sagemaker_mp_enabled,
    # get_parameter_names,
    # has_length,
    ALL_LAYERNORM_LAYERS,
    # logger,
)

logger = logging.get_logger(__name__)



class OpenXTrainer(Trainer):
    def __init__(self, *args, eval_dataset_fixed_traj=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.eval_dataset_fixed_traj = eval_dataset_fixed_traj


        self.text_table = wandb.Table(columns=["eval_loss", 'eval_rouge1',
                                           'eval_images',
                                           'eval_decoded_preds', 'eval_decoded_labels',
                                           'eval_token_preds', 'eval_token_labels'])


    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        if not hasattr(self, 'eval_ct'):
            self.eval_ct = 1
        else:
            self.eval_ct += 1

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        device = self.model.lm.device if hasattr(self.model, 'lm') else self.model.vlm_model.device

        if hasattr(self, 'only_vis_eval') and self.only_vis_eval == True:
            try:
                print("Start evaluatign on fixed trajectory")
                if self.eval_dataset_fixed_traj is not None:
                    dataloader_eval_dataset_fixed_traj = self.accelerator.prepare(DataLoader(
                        self.eval_dataset_fixed_traj))  # , num_workers=4))

                    # ds  = self.accelerator.prepare(self.eval_dataset_fixed_traj)
                    print("Evaluate on complete fixed trajectory and visualize")
                    eval_dir = os.path.join(self.args.output_dir, "eval_results", f"chkpt_{self.state.global_step}")
                    os.makedirs(eval_dir, exist_ok=True)
                    for step, traj in enumerate(dataloader_eval_dataset_fixed_traj):
                        try:
                            plot_identifier = str(traj['seq_nr'].item()).zfill(4)
                            evaluate_on_fixed_trajectory(self.model, hist_len=self.args.fixed_traj_eval_hist_len,
                                                         traj=traj,
                                                         name=plot_identifier, eval_result_dir=eval_dir,
                                                         from_data_loader=True)
                        except Exception as e:
                            print(traceback.format_exc())

            except Exception as e:
                print(traceback.format_exc())

            import sys; sys.exit()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        example_images = []

        # action_acc_list, none_action_ratio_list = [], []
        all_action_acc = torch.zeros((0, 7), device=device)
        all_action_acc_no_none = torch.zeros((0, 7), device=device)
        none_action_ratio_list = None

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            if observed_num_examples == 0 and 'image' in inputs.keys():
                example_images = inputs['image'][:3]
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            meta_info = {'eval_ct': self.eval_ct, 'seq_nr': step}

            action_acc, action_acc_no_none, none_action_ratio = self.eval_ar_generation(model, inputs, inputs["labels"], meta_info)


            # print("action accuracy, gpu", device)
            # print(action_acc)
            # print(all_action_acc)
            # print("action accuracy no none, gpu", device)
            # print(action_acc_no_none)
            # print(all_action_acc_no_none)

            action_acc_no_none = torch.tensor(action_acc_no_none, device=device).reshape((-1, 7))
            action_acc_no_none = self.accelerator.gather_for_metrics((action_acc_no_none))
            all_action_acc_no_none = torch.cat((all_action_acc_no_none, action_acc_no_none))

            action_acc = torch.tensor(action_acc, device=device).reshape((-1, 7))
            action_acc = self.accelerator.gather_for_metrics((action_acc))
            all_action_acc = torch.cat((all_action_acc, action_acc))
            # all_action_acc = action_acc if action_acc_list is None else torch.cat((action_acc_list, action_acc))

            print("Action ratio list, gpu", device)
            # print(len(none_action_ratio_list))
            print(none_action_ratio_list)
            none_action_ratio = torch.tensor(none_action_ratio, device=device).reshape((-1, 1))
            none_action_ratio = self.accelerator.gather_for_metrics((none_action_ratio))
            none_action_ratio_list = none_action_ratio if none_action_ratio_list is None else torch.cat(
                (none_action_ratio_list, none_action_ratio))
            # action_acc_list.extend(self.accelerator.gather_for_metrics((action_acc)))
            # none_action_ratio_list.extend(self.accelerator.gather_for_metrics(([none_action_ratio])))

            if not hasattr(self, 'skip_loss_eval') or not self.skip_loss_eval:
                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only,
                                                            ignore_keys=ignore_keys)

                main_input_name = getattr(self.model, "main_input_name", "input_ids")
                inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

                # Update containers on host
                if loss is not None:
                    losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                    losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
                if labels is not None:
                    labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                if inputs_decode is not None:
                    inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                    inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                    inputs_host = (
                        inputs_decode
                        if inputs_host is None
                        else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                    )
                if logits is not None:
                    logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                    if self.preprocess_logits_for_metrics is not None:
                        logits = self.preprocess_logits_for_metrics(logits, labels)
                    logits = self.accelerator.gather_for_metrics((logits))
                    preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

                if labels is not None:
                    labels = self.accelerator.gather_for_metrics((labels))
                    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)


                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                # if args.eval_accumulation_steps is not None and self.accelerator.sync_gradients:
                if step % args.eval_accumulation_steps == 0:
                    if losses_host is not None:
                        losses = nested_numpify(losses_host)
                        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                    if preds_host is not None:
                        logits = nested_numpify(preds_host)
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    if inputs_host is not None:
                        inputs_decode = nested_numpify(inputs_host)
                        all_inputs = (
                            inputs_decode
                            if all_inputs is None
                            else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                        )
                    if labels_host is not None:
                        labels = nested_numpify(labels_host)
                        all_labels = (
                            labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                        )

                    # Set back to None to begin a new accumulation
                    losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)



        # wandb_images = [wandb.Image(img.permute(1,2,0).cpu().numpy(), caption=f'{i}') for i, img in enumerate(example_images)]
        #
        # self.text_table.add_data(metrics['eval_loss'], metrics['eval_rouge1'],
        #                          wandb_images,
        #                          metrics['eval_decoded_preds'], metrics['eval_decoded_labels'],
        #                          metrics['eval_token_preds'], metrics['eval_token_labels'])
        #
        # # bug in wandb, need to create new table every time we log, see https://github.com/wandb/wandb/issues/2981#issuecomment-988794468
        # self.text_table = wandb.Table(
        #     columns=self.text_table.columns, data=self.text_table.data
        # )
        # # self.log(metrics)
        # # self.log({'table': self.text_table})
        # # self.log({'table2': new_table})
        # metrics['eval_table'] = self.text_table
        #
        # for key in ['eval_decoded_preds', 'eval_decoded_labels', 'eval_token_preds', 'eval_token_labels']:
        #     metrics.pop(key)

        if all_action_acc.shape[0] > 0: # is not None:
            # per_action_dim_acc = np.mean(np.stack(action_acc_list, axis=0), axis=0)
            all_action_acc = all_action_acc[torch.where(all_action_acc > -1000)].reshape((-1, 7))
            per_action_dim_acc = all_action_acc.mean(0)
            metrics['eval_action_actions_accuracy_mean'] = torch.mean(per_action_dim_acc).item()
            for i, acc in enumerate(per_action_dim_acc):
                metrics[f"eval_action_actions_accuracy_dim_{i}"] = per_action_dim_acc[i].item()
        if all_action_acc_no_none.shape[0] > 0: # is not None:
            # per_action_dim_acc = np.mean(np.stack(action_acc_list, axis=0), axis=0)
            all_action_acc_no_none = all_action_acc_no_none[torch.where(all_action_acc_no_none > -1000)].reshape((-1, 7))
            per_action_dim_acc = all_action_acc_no_none.mean(0)
            metrics['eval_action_actions_accuracy_no_none_mean'] = torch.mean(per_action_dim_acc).item()
            for i, acc in enumerate(per_action_dim_acc):
                metrics[f"eval_action_actions_accuracy_no_none_dim_{i}"] = per_action_dim_acc[i].item()

        # metrics['eval_action_none_actions_ratio'] = sum(none_action_ratio_list) / len(none_action_ratio_list)
        metrics['eval_action_none_actions_ratio'] = none_action_ratio_list.mean().item()

        try:
            print("Start evaluating on fixed trajectory")
            if self.eval_dataset_fixed_traj is not None:
                dataloader_eval_dataset_fixed_traj = self.accelerator.prepare(DataLoader(
                    self.eval_dataset_fixed_traj))#, num_workers=4))

                # ds  = self.accelerator.prepare(self.eval_dataset_fixed_traj)
                print("Evaluate on complete fixed trajectory and visualize")
                eval_dir = os.path.join(self.args.output_dir, "eval_results", f"chkpt_{self.state.global_step}")
                os.makedirs(eval_dir, exist_ok=True)
                for step, traj in enumerate(dataloader_eval_dataset_fixed_traj):
                    try:
                        plot_identifier = str(traj['seq_nr'].item()).zfill(4)
                        evaluate_on_fixed_trajectory(self.model, hist_len=self.args.fixed_traj_eval_hist_len, traj=traj,
                                                     name=plot_identifier, eval_result_dir=eval_dir, from_data_loader=True)
                    except Exception as e:
                        print(traceback.format_exc())

        except Exception as e:
            print(traceback.format_exc())



        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)

    @torch.no_grad()
    def eval_ar_generation(self, model, inputs, labels, meta_info, print_info=False):
        # generated_ids, decoded_pred_actions = model.generate(inputs['instruction'], inputs['images'], meta_info, return_decoded_actions=True)

        generated_ids, decoded_pred_actions = model.generate(
            inputs['instruction'], inputs['images'], meta_info, return_decoded_actions=True,
            gt_history=True, image_pos=inputs['image_pos'], text=inputs['text'],
            action_pos=inputs['action_pos'] if 'action_pos' in inputs.keys() else None,
            qformer_input_ids=inputs['qformer_input_ids'] if 'qformer_input_ids' in inputs.keys() else None)

        pad_token_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.img_pad_id
        label_str = model.tokenizer.batch_decode(torch.where(labels != -100, labels, pad_token_id),
                                                 skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            gen_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            instr_str = model.tokenizer.batch_decode(inputs['instruction'], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            # print(generated_ids)
            # print(labels)
            if print_info:
                print(f"###### Generated text for input: {instr_str}######")
                print(gen_text)
                print(f"###### Label text for input: {instr_str}######")
                print(label_str)
                print("############################################")
        except:
            pass

        none_actions = 0
        action_diff_no_none = np.zeros((0, 7))
        action_diff = np.zeros((0, 7))

        try:
            gt_actions = self.extract_action_from_label(labels, label_str, inputs, model)
            decoded_gt_actions = [text_to_action(da, gripper_range_2=True)  for da in gt_actions] #TODO gripper range)

            #TODO remove [ea] from text (is included in the text to type)


            # success = []
            for i in range(len(decoded_gt_actions)):
                gta = decoded_gt_actions[i]
                if len(decoded_pred_actions) <= i:
                    none_actions += 1
                    continue
                pra = decoded_pred_actions[i]
                # pra = [x - 0.01 for x in gta] #TODO for debugging only
                print("GT   action", gta)
                print("Pred action", pra)


                if pra is None:
                    none_actions += 1
                    action_diff = np.concatenate((action_diff, np.ones((1, 7)) + 1), axis=0) # 2 is the max posiible error per dim
                    continue
                action_diff_no_none = np.concatenate((action_diff_no_none,
                                              np.abs(np.array(gta) - np.array(pra)).reshape((1,7))),
                                             axis=0)
                action_diff = np.concatenate((action_diff,
                                              np.abs(np.array(gta) - np.array(pra)).reshape((1,7))),
                                             axis=0)


                    # except Exception as e:
                    #     print(f"Error in action checking. gta: {gta} | pra: {pra}")
                    #     success.append(False)

            #     print("success: ", success[-1])
            #
            # print("Action prediction accuracy: ", sum(success) / len(success))

        except Exception as e:
            print(e)
            print(''.join(traceback.TracebackException.from_exception(e).format()))
            # return None, 1 #TODO what to do in this case

        if action_diff.shape[0] > 0:
            action_diff = np.mean(action_diff, axis=0)
        else:
            action_diff = np.zeros(7) - 10000 # to filter out, o/w with empty tensor error in gathering from different devices
        if action_diff_no_none.shape[0] > 0:
            action_diff_no_none = np.mean(action_diff_no_none, axis=0)
        else:
            action_diff_no_none = np.zeros(7) - 10000 # to filter out, o/w with empty tensor error in gathering from different devices
        # else:
        #     action_diff = None

        print(none_actions, len(decoded_gt_actions))
        none_actions_ratio = none_actions / len(decoded_gt_actions)
        return action_diff, action_diff_no_none, none_actions_ratio


    def extract_action_from_label(self, labels, label_str, inputs, model):
    # def extract_action_from_label(labels, label_str, image_pos, model):
        action_list = []
        # if '\n' in label_str:
        if "Qwen" in model.image_model_name:
            # action_start_ids = [m.end() for m in re.finditer(': \n', label_str)]
            action_start_ids = [0] + [m.end() for m in re.finditer('\[ea\]', label_str)]
            for i in range(len(action_start_ids)-1):
                action_region = label_str[action_start_ids[i]:action_start_ids[i+1]]
                action = action_region.partition('[ea]')[0]
                action_list.append(action)
            # action_list.append(label_str[action_start_ids[-1]:].partition('[ea]')[0])
            decoded_action_list = action_list

        if "llava" in model.image_model_name:
            # action_start_ids = [m.end() for m in re.finditer(': \n', label_str)]
            action_start_ids = [0] + [m.end() for m in re.finditer('\[ea\]', label_str)]
            for i in range(len(action_start_ids)-1):
                action_region = label_str[action_start_ids[i]:action_start_ids[i+1]]
                action = action_region.partition('[ea]')[0].partition('ASSISTANT: ')[-1]
                action_list.append(action)
            # action_list.append(label_str[action_start_ids[-1]:].partition('[ea]')[0])
            decoded_action_list = action_list

        if "blip" in model.image_model_name:
            for i in range(inputs['action_pos'].shape[-1]):
                st, end = inputs['action_pos'][0,:,i]
                action = labels[0, st:end]
                action_list.append(action)

            decoded_action_list = model.tokenizer.batch_decode(action_list,
                                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)

        else:
            image_pos = inputs['image_pos']
            for i in range(image_pos.shape[1] - 1):
                action_list.append(labels[0, image_pos[0, i]+1:image_pos[0, i+1]])
            action_list.append(labels[0, image_pos[0, -1]+1:])

            decoded_action_list = model.tokenizer.batch_decode(action_list,
                                                     skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return decoded_action_list


class LLaVATrainer(OpenXTrainer):

    # def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
    #     if self.train_dataset is None or not has_length(self.train_dataset):
    #         return None
    #
    #     if self.args.group_by_modality_length:
    #         lengths = self.train_dataset.modality_lengths
    #         return LengthGroupedSampler(
    #             self.args.train_batch_size,
    #             world_size=self.args.world_size * self.args.gradient_accumulation_steps,
    #             lengths=lengths,
    #             group_by_modality=True,
    #         )
    #     else:
    #         return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

