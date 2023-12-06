from abc import ABC

import numpy as np
import torch
from torch import tensor as tt

from gyms.maskcrn.models.tokens import TokenSeq
from gyms.maskcrn.models.utils import to_numpy
MASKED_VALUE = 0

class MASKING(ABC):

    RTG_MASKING_TYPES = [
        "fixed_first",
        "Unchanged",
    ]

    def __init__(
        self,
        input_data,
        rtg_masking_type,
        silent=False,
        **kwargs,
    ):
        self.input_data = input_data
        self.mask_shape = input_data.shape
        self.rtg_masking_type = rtg_masking_type

        self.seq_len = self.input_data.seq_len
        self.num_seqs = self.input_data.num_seqs

        self.silent = silent

        # This has to be computed in the subclasses' init methods
        self.input_masks = None
        self.prediction_masks = None

        self.loss = None

        self.computed_output = False

    @classmethod
    def from_params(cls, input_data, batch_params):
        batch_class = batch_params["type"]
        return batch_class(input_data=input_data, **batch_params)

    def get_input_masks(self):
        """Get the input masks for observations and feeds. All logic will be in subclasses"""
        raise NotImplementedError()

    @staticmethod
    def postprocess_rtg_mask(rtg_mask, rtg_masking_type = "fixed_first"):
        """
        Various kinds of rtg masking.

        - fixed_first: reward-conditioning without randomization (first rtg tokens always present, rest always masked).
        - Unchanged: just keep whatever masking scheme the batch type generates
        """
        assert rtg_masking_type in MASKING.RTG_MASKING_TYPES

        num_seqs, seq_len = rtg_mask.shape
        if rtg_masking_type == "fixed_first":
            rtg_mask[:, 0] = 1
            rtg_mask[:, 1:] = 0

        elif rtg_masking_type == "Unchanged":
            pass

        else:
            raise ValueError("rtg_masking_type not recognized")

        return rtg_mask

    def get_prediction_masks(self):
        """By default, predict everything that wan't present in the input"""
        s_mask = 1 - self.input_masks["*"]["state"]
        a_mask = 1 - self.input_masks["*"]["action"]
        r_mask = 1 - self.input_masks["*"]["rtg"]
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return 1

    def num_maskings_per_type(self):
        """
        If the batch allows for more than one masking type, we want to be able to perfectly tile N maskings of each
        type in the batch in order to reduce variance. We use `must_have_size_multiple_of` to determine how many
        sequences we should mask with each masking type.
        """
        num_masking_types = self.must_have_size_multiple_of(self.seq_len)
        assert (
            self.num_seqs % num_masking_types == 0
        ), "Num seqs in the batch {} must be divisible by num_masking_types {}".format(self.num_seqs, num_masking_types)
        num_per_type = self.num_seqs // num_masking_types
        return num_per_type

    @property
    def model_input(self):
        inp, timestep_inp = self.input_data.model_input(self.input_masks)
        return inp, timestep_inp

    def empty_input_masks(self):
        s_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        act_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        rtg_in_mask = torch.zeros((self.num_seqs, self.seq_len))
        return act_in_mask, rtg_in_mask, s_in_mask

    def empty_pred_masks(self):
        s_mask = torch.zeros_like(self.input_masks["*"]["state"])
        a_mask = torch.zeros_like(self.input_masks["*"]["action"])
        r_mask = torch.zeros_like(self.input_masks["*"]["rtg"])
        return a_mask, r_mask, s_mask

    def get_factor(self, factor_name):
        return self.input_data.get_factor(factor_name)

    def get_input_mask_for_factor(self, factor_name):
        if "*" in self.input_masks:
            mask_key = TokenSeq.get_mask_key(factor_name, self.input_masks["*"])
            return self.input_masks["*"][mask_key]
        else:
            raise NotImplementedError()

    def get_masked_input_factor(self, factor_name, mask_nans=False):
        factor = self.get_factor(factor_name)
        input_mask = self.get_input_mask_for_factor(factor_name)
        return factor.mask(input_mask, mask_nans)

    def get_prediction_mask_for_factor(self, factor_name):
        if "*" in self.prediction_masks:
            return self.prediction_masks["*"][factor_name]
        else:
            raise NotImplementedError()

    ###################

    def add_model_output(self, batch_output):
        self.input_data.add_model_output(batch_output)
        self.computed_output = True

    def compute_loss_and_acc(self, loss_weights):
        """
        We are given the output of the transformer
        We now want to computing losses and accuracy directly on a output head (predicting in behaviour space)

        NOTE: currently does not do accuracies
        """
        assert self.computed_output, "Have to add output with add_model_output before trying to compute loss"

        # need to implement a get_masked_items for Factors
        loss_dict = self.input_data.get_loss(loss_weights, self.prediction_masks)

        total_loss = 0.0
        for ts_name, factors_dict in loss_dict.items():
            for factor_name, v in factors_dict.items():
                total_loss += v.cpu()

        self.loss = total_loss
        loss_dict["total"] = total_loss
        return loss_dict

    @classmethod
    def get_dummy_batch_output(cls, data, batch_params, trainer):
        """
        Based on some data (in FullTokenSeq format), create a dummy batch and return it with the computed predictions

        TODO: have a parameter to do this with the model in eval mode, so as to not accidentally not use eval mode when
         evaluating
        """
        b = cls.from_params(data, batch_params)
        trainer.model(b)
        return b



class ContextSeg(MASKING):
    """
    Mask which has 1..100..001..1 for actions
    and            1..110..001..1 for states

    where 1s are not masked out, and 0s are masked out. You can sample actions from the back _or_ from the front.

    For span_limit (a,b), will mask out an additional action at the beginning to predict:
        States [a, b) and actions [a-1, b) will be masked out (unless a=0 for first-timestep backwards
        inference, actions [a, b)).
    """

    def __init__(self, seg_limits=None, **kwargs):
        super().__init__(**kwargs)

        # Masked span to predict at inference time
        # None if training (masks)
        self.seg_limits = seg_limits
        self.training_seg = self.get_training_seg(self.seq_len)
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

        if self.seg_limits is not None:
            assert type(seg_limits) in [tuple, list]
            seg_limits = tuple(seg_limits)
            assert self.seq_len >= seg_limits[1] >= seg_limits[0] >= 0, (
                self.seq_len,
                seg_limits[1],
                seg_limits[0],
            )
            assert seg_limits in self.training_seg, f"{seg_limits}vs{self.training_seg}"

    @staticmethod
    def get_training_seg(seq_len):
        possible_seg_limits = []
        for start in np.arange(seq_len + 1):
            for end in np.arange(start, seq_len + 1):
                possible_seg_limits.append((start, end))
        return possible_seg_limits

    @classmethod
    def must_have_size_multiple_of(cls, seq_len):
        """Batch should have a number of sequences multiple of the returned number"""
        return len(cls.get_training_seg(seq_len))

    def generate_training_seg_limits(self):
        """Tiles all possible training span combinations to get `self.num_seqs` training masks."""
        num_per_type = self.num_maskings_per_type()
        training_seg = self.get_training_seg(self.seq_len)
        seg_limits_n = np.concatenate([[span_limits] * num_per_type for span_limits in training_seg])
        np.random.shuffle(seg_limits_n)
        return seg_limits_n

    def get_masks_for_seg_limits(self, span_limits_n):
        """
        Masks state, ac, rtg tensors according to `span_limits`.
        Each token sequence will be of shape [num_seqs, seq_len, factor_size_sum]
        """

        s_in_mask = torch.ones((self.num_seqs, self.seq_len))
        act_in_mask = torch.ones((self.num_seqs, self.seq_len))
        rtg_in_mask = torch.ones((self.num_seqs, self.seq_len))

        for traj_idx, span_limits in enumerate(span_limits_n):
            # Mask out the things we want to predict (i.e. zero out)
            a, b = span_limits
            # mask from the start of the sequence to have different sequence length
            s_in_mask[traj_idx, a:b] = 0
            rtg_in_mask[traj_idx, a:b] = 0

            a_in_start_idx = max(a - 1, 0)
            act_in_mask[traj_idx, a_in_start_idx:b] = 0
            act_in_mask[traj_idx, a_in_start_idx:b] = 0

        rtg_in_mask = self.postprocess_rtg_mask(rtg_in_mask, self.rtg_masking_type)
        return {"*": {"state": s_in_mask, "action": act_in_mask, "rtg": rtg_in_mask}}

    def get_input_masks(self):
        """Get masked s, a, rtg for training or inference."""
        # Inference: predict with a particular span masked
        if self.seg_limits is not None:
            seg_limits_n = [self.seg_limits] * self.num_seqs
        # Training: generate training masks
        else:
            seg_limits_n = self.generate_training_seg_limits()

        return self.get_masks_for_seg_limits(seg_limits_n)


class FuturePred(ContextSeg):
    """
    later states/actions are all zeros always.
    """

    @staticmethod
    def get_training_seg(seq_len):
        possible_span_limits = []
        for start in np.arange(1, seq_len + 1):
            end = seq_len
            possible_span_limits.append((start, end))
        return possible_span_limits



class NextActionPred(FuturePred):
    """
    predict the next action (instead of all future states and actions).
    
    Val loss will be average loss on p(a_t | s_1, a_1, ... s_{t-1}, a_{t-1}), *for all t*.
    """

    def get_prediction_masks(self):
        s_mask, a_mask, r_mask = self.empty_pred_masks()

        # Only look at prediction loss for first missing action
        # First missing action for each seq in the batch
        first_missing_act_to_pred_t_n = self.input_masks["*"]["action"].sum(1).long()
        for seq_idx, first_missing_t in enumerate(first_missing_act_to_pred_t_n):
            a_mask[seq_idx, first_missing_t] = 1

        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}



class RandomMask(NextActionPred):
    @staticmethod
    def get_training_seg(seq_len):
        # Differs from the super() method in that the start index cannot be the last state
        possible_span_limits = []
        for start in np.arange(1, seq_len):
            end = seq_len
            possible_span_limits.append((start, end))
        return possible_span_limits

    def generate_training_span_limits(self):
        # Differs from super in that that span limits is randomized rather than tessellated
        training_spans = self.get_training_seg(self.seq_len)
        span_indices = np.random.choice(len(training_spans), size=self.num_seqs)
        span_limits_n = [training_spans[idx] for idx in span_indices]
        return span_limits_n

    def get_input_masks(self):
        input_masks = super().get_input_masks()
        # Add goal to seen
        input_masks["*"]["state"][:, self.seq_len - 1] = 1
        return input_masks



class RandomLengthMask(ContextSeg):
    """
    Random Mask to generate different token length
    rtg masking is fixed_first: first rtg tokens always present, rest always masked.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_masks = self.get_input_masks()
        self.prediction_masks = self.get_prediction_masks()

    def get_masked_rtg(returns_to_go, rtg_mask):
        rtgs = returns_to_go.clone()
        masked_input = rtgs.float()
        masked_input[rtg_mask == 0] = MASKED_VALUE
        return masked_input

    def get_input_masks_state_action(num_seq_1, seq_lens):

        return 0

    def get_input_masks(self):
        mask_size = (self.num_seqs, self.seq_len)
        s_mask = np.ones(mask_size)
        a_mask = np.ones(mask_size)
        rtg_mask = np.ones(mask_size)
        rtg_mask = self.postprocess_rtg_mask(rtg_mask, self.rtg_masking_type)
        return {"*": {"state": tt(s_mask), "action": tt(a_mask), "rtg": tt(rtg_mask)}}

    def get_prediction_masks(self):
        s_mask = torch.zeros_like(self.input_masks["*"]["state"])
        a_mask = torch.ones_like(self.input_masks["*"]["action"])
        r_mask = torch.zeros_like(self.input_masks["*"]["rtg"])
        return {"*": {"state": s_mask, "action": a_mask, "rtg": r_mask}}

