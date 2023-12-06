
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor as tt

from gyms.maskcrn.models import utils
from gyms.maskcrn.models.utils import to_numpy


AVG_METRICS = False
MASKED_VALUE = 0
MISSING_VALUE = np.nan


class FactorSeq:
    """
    factor: "state", "action", etc.

    In terms of the data, for a generic single-agent RL setup we would expect shapes such as:
    - States will be [num_trajs, state_shape, seq_len]
    - Actions will be [num_trajs, action_shape, seq_len]
    - Rewards will be [num_trajs, reward_shape, seq_len]
    """

    def __init__(
        self,
        name,
        input_data,
        loss_type,
        output=None,
        output_mask=None,
        input_mask=None,
        device='cuda',
        **kwargs,
    ):
        assert isinstance(input_data, torch.Tensor)
        self.name = name
        self.input = input_data
        self.loss_type = loss_type
        self.shape = self.input.shape
        self.num_seqs, self.seq_len, self.size = self.shape
        assert len(self.input.shape) == 3, self.input.shape
        assert self.num_seqs > 0, "Making an empty factor is probably a bug?"

        # Output
        self.output = output
        # Mask that was used to generate the output
        self.output_mask = output_mask


    def __getitem__(self, key):
        out, out_m, in_m = None, None, None

        if self.output is not None:
            assert self.output_mask is not None
            # assert self.input_mask is not None
            out = self.output.__getitem__(key)
            out_m = self.output_mask.__getitem__(key)
            # in_m = self.input_mask.__getitem__(key)

        return self.__class__(
            name=self.name,
            input_data=self.input.__getitem__(key),
            loss_type=self.loss_type,
            output=out,
            output_mask=out_m,
            input_mask=in_m,
        )

    @property
    def missing_ts(self):
        if hasattr(self, "_missing_ts"):
            return self._missing_ts

        if self.loss_type == "sce":
            max_vals = self.input.max(axis=2)[0]
            missing_inputs = max_vals.isnan()
        elif self.loss_type == "l2":
            missing_inputs = self.input.isnan()
            assert (missing_inputs.any(dim=2) == missing_inputs.all(dim=2)).all(), (
                "If a part of a factor is nan, the" "entire factor should be nan"
            )
            missing_inputs = missing_inputs.any(dim=2)
        else:
            raise NotImplementedError("Loss type {} not recognized for Factor {}".format(self.loss_type, self.name))

        # Check same missing ts across sequences
        assert (
            missing_inputs.any(dim=0) == missing_inputs.all(dim=0)
        ).all(), "If one input sequence has a specific missing input, all the other ones should have the same"
        missing_inputs = missing_inputs.any(dim=0)
        self._missing_ts = [t for t, missing in enumerate(missing_inputs) if missing]
        return self._missing_ts

    def mask(self, mask, mask_nans=False):
        """
        Will mask the input.

        When missing inputs exist, we want to assert that they are being masked at model input time
        """
        return self.mask_data(self.input, mask, mask_nans)

    @staticmethod
    def mask_data(data, mask, mask_nans=False):
        """
        Will mask the input.

        When missing inputs exist, we want to assert that they are being masked at model input time
        """
        assert mask.shape[:2] == data.shape[:2], f"{mask.shape} vs {data.shape}"
        assert len(mask.shape) in [len(data.shape), len(data.shape) - 1]

        # mask will be [num_seqs, seq_len]
        # self.data will be [num_seqs, seq_len, factor_size]
        masked_input = data.clone()
        masked_input[mask == 0] = MASKED_VALUE
        if mask_nans:
            masked_input[masked_input.isnan()] = MASKED_VALUE
        else:
            assert (
                not masked_input.isnan().any()
            ), f"Some Factor: Mask does not match NaN masking in data, check whether timesteps have been updated in the input"
        return masked_input

    def add_model_output(self, output_data):
        assert output_data.shape == self.input.shape
        assert self.output is None, "You shouldn't be predicting twice for the same set of data"
        self.output = output_data

    def is_structurally_equal(self, other):
        """
        Returns whether other instance of FactorSeq has the same structure (even though it might have different
        number of sequences and sequence contents). Used for checking e.g. whether two instances can be merged
        """
        assert isinstance(other, FactorSeq)
        assert self.__dict__.keys() == other.__dict__.keys()

        # The inputs are allowed to be different. Number of sequences also, so shape should also be ignored.
        keys_to_ignore = ["input", "num_seqs", "shape"]
        # NOTE: timestep will always have loss_weight: np.nan, which always returns false if you check for equality with itself
        if self.name == "timestep":
            keys_to_ignore.append("loss_weight")
        return dict_equal_check(self, other, keys_to_ignore)

    def merge(self, other):
        """Returns a new instance with data from both cases"""
        assert self.is_structurally_equal(other)
        merged_data = torch.cat([self.input, other.input], dim=0)
        assert self.output is None, "For now merging does not support post-output merging"
        return self.__class__(
            name=self.name,
            input_data=merged_data,
            loss_type=self.loss_type,
            output=None,
            output_mask=None,
            input_mask=None,
        )

    @classmethod
    def concatenate(cls, factors):
        assert all(isinstance(f, FactorSeq) for f in factors)
        assert all(factors[0].is_structurally_equal(f) for f in factors)
        assert all(f.output is None for f in factors), "For now merging does not support post-output merging"
        merged_data = torch.cat([f.input for f in factors], dim=0)
        return cls(
            name=factors[0].name,
            input_data=merged_data,
            loss_type=factors[0].loss_type,
            output=None,
            output_mask=None,
            input_mask=None,
        )

    @property
    def inputs_hr(self):
        """
        The inputs in a human readable format. For discretized Factors, this is equivelent to taking the argmax of
        the one hot encoding.
        """
        # NOTE: maybe come up with better name for this. Really what it is a non one-hot version of the inputs.
        if self.loss_type == "sce":
            argmax = self.input.argmax(dim=2).float()
            argmax[self.input.isnan().any(dim=2)] = MISSING_VALUE
            if not argmax.isnan().any():
                argmax = argmax.int()
            return argmax
        else:
            return self.input.squeeze()

    @property
    def predictions(self):
        if self.loss_type == "sce":
            return self.output.argmax(dim=2)
        else:
            raise NotImplementedError()

    @property
    def prediction_probs(self):
        if self.loss_type == "sce":
            return to_numpy(torch.nn.Softmax(dim=2)(self.output))
        else:
            raise NotImplementedError()

    def sample_timestep_predictions(self, num_samples, t, argmax=False):
        if self.loss_type == "sce":
            sample_probs = self.prediction_probs[:, t]
            if argmax:
                assert num_samples == 1, "Why take more than 1 if the same?"
                samples = torch.argmax(tt(sample_probs), dim=-1)
            else:
                samples = torch.multinomial(tt(sample_probs), num_samples=num_samples)
            return samples, sample_probs
        else:
            raise NotImplementedError()

    def get_loss(self, loss_weight, pred_mask=None):
        """
        Computes the loss for the predictions for the current FactorSeq

        This should only be done once, as computing the loss also saves the mask
        """
        if pred_mask is None:
            assert self.output_mask is not None
            pred_mask = self.output_mask
        else:
            self.output_mask = pred_mask.to(int)

        assert pred_mask.shape == self.output.shape[:2], "{} vs {}".format(pred_mask.shape, self.output.shape)

        # pred_mask: [num_trajs, seq_len]
        pred_mask = pred_mask.unsqueeze(2)

        preds = get_masked_items(self.output, pred_mask)
        targets = get_masked_items(self.input, pred_mask).float()
        assert preds.shape == targets.shape

        if len(preds) == 0 or loss_weight == 0:
            # A torch 0, to make it consistent with other loss values which will be tensors
            return tt([0])[0]

        if self.loss_type == "l2":
            loss = nn.MSELoss()(preds, targets)

        elif self.loss_type == "sce":
            # Targets are passed in a one-hot-encoding and have to be flattened
            assert self.missing_ts == [], (
                "You probably shouldn't be computing losses for inputs with missing timesteps. "
                "You don't have targets for them!"
            )
            assert (targets.sum(dim=1) == 1).all(), "{}".format(targets.sum(dim=1) == 1)
            targets = targets.argmax(dim=1)
            loss = nn.CrossEntropyLoss()(preds, targets)
        else:
            raise NotImplementedError()

        return loss * loss_weight

    def get_accuracy(self, mode="avg"):
        assert self.loss_type == "sce", "Accuracy only makes sense in discretized settings"

        targets = self.input.argmax(2)
        predictions = self.output.argmax(2)

        matches = (targets == predictions).to(int)
        acc_by_t = []
        num_by_t = tt([self.output_mask[:, t].sum() for t in range(self.seq_len)])
        for t in range(self.seq_len):
            t_acc = get_masked_items(matches[:, t], self.output_mask[:, t]).to(float).mean()
            acc_by_t.append(t_acc)

        if mode == "avg":
            # Remove timesteps that don't have any predictions from the calculation
            acc_by_t, num_by_t = zip(*[(n, a) for n, a in zip(num_by_t, acc_by_t) if n != 0])
            weight_by_t = tt(num_by_t) / tt(num_by_t).sum()
            return (tt(acc_by_t) * weight_by_t).sum()
        elif mode == "by_t":
            return tt(acc_by_t), num_by_t
        else:
            raise ValueError()


class TokenSeq:
    """
    A factor group is a set of Factors that will be fed into the transformer together (will receive only 1 joint
    embedding). I.e., they will form 1 token.
    """

    def __init__(self, name, factors):
        assert all([isinstance(factor, FactorSeq) for factor in factors])
        self.name = name
        self.factors = factors

        # NOTE: the timestep factor is treated as a special case throughout the code, as
        #  it will not be fed to the model. Rather it will be used as an alternative to the positional encoding
        self.factor_sizes = tt([factor.size for factor in self.factors if factor.name != "timestep"])
        self.factor_names = [f.name for f in self.factors]
        assert len(self.factor_names) == len(set(self.factor_names)), "factor names must be unique"

        # Check factors have same num of trajectories, and of same sequence length
        assert all([self.factors[0].num_seqs == factor.num_seqs for factor in self.factors])
        assert all([self.factors[0].seq_len == factor.seq_len for factor in self.factors])

        self.num_seqs = self.factors[0].num_seqs
        self.seq_len = self.factors[0].seq_len
        self.input = self.model_input()
        self.shape = self.input.shape

    def __repr__(self):
        return str([(f.name, f.shape) for f in self.factors])

    def is_structurally_equal(self, other):
        """
        Returns whether other instance of TokenSeq has the same structure (even though it might have different
        number of sequences). Useful for checking e.g. whether two instances can be merged
        """
        assert isinstance(other, TokenSeq)
        assert self.__dict__.keys() == other.__dict__.keys()
        # The factor seqs are checked separately. Number of sequences are allowed to be different, so shape should also be ignored.
        # Input may have nans (which will never count as equal), and are checked separately within factors, so should also be ignored.
        keys_to_ignore = ["factors", "num_seqs", "shape", "input"]
        factors_eq = [ts0.is_structurally_equal(ts1) for ts0, ts1 in zip(self.factors, other.factors)]
        return dict_equal_check(self, other, keys_to_ignore) and all(factors_eq)

    def merge(self, other):
        """Returns a new instance with data from both cases"""
        assert self.is_structurally_equal(other)
        fs_n = []
        for fs0, fs1 in zip(self.factors, other.factors):
            fs_n.append(fs0.merge(fs1))
        return self.__class__(name=self.name, factors=fs_n, loss_weight=self.loss_weight)

    @classmethod
    def concatenate(cls, token_seqs):
        assert all(isinstance(t, TokenSeq) for t in token_seqs)
        assert all(token_seqs[0].is_structurally_equal(t) for t in token_seqs)
        concatenated_factors_n = []
        for factors in zip(*[t.factors for t in token_seqs]):
            concatenated_factors_n.append(FactorSeq.concatenate(factors))
        return cls(name=token_seqs[0].name, factors=concatenated_factors_n)

    @classmethod
    def add_model_output(cls, output, input_token_seq):
        start_idx = 0
        for factor in input_token_seq.factors:
            # The timestep will not be predicted, as is not fed into the network directly (and thus also not outputted)
            if factor.name == "timestep":
                continue
            end_idx = start_idx + factor.size
            factor.add_model_output(output[:, :, start_idx:end_idx])
            start_idx = end_idx

    @staticmethod
    def get_mask_key(factor_name, mask_dict):
        """
        Given a factor and a mask dict, determine which mask it should be using.

        NOTE: Quite hacky right now. Say you want to have multiple components of the state,
         e.g. state_pos, state_door, etc.
         As long as you name them like this (with an underscore), all masking should work smoothly.
         The current masks that are defined are only for things that look like "action_X", "state_X", "rtg_X"

        TODO: add clear documentation about this somewhere
        """
        if factor_name.split("_")[0] in mask_dict.keys():
            mask_key = factor_name.split("_")[0]
        else:
            mask_key = "*"
        assert mask_key in mask_dict.keys(), f"Key {factor_name} was not among keys for mask_dict: {mask_dict.keys()}"
        return mask_key

    def mask_factors(self, input_mask_dict=None):
        # NOTE: Don't include timestep factor in the return call when querying for masked factors
        if input_mask_dict is None:
            return [f.input for f in self.factors if f.name != "timestep"]

        masked_factors = []
        for f in self.factors:
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, input_mask_dict)
            masked_factors.append(f.mask(input_mask_dict[mask_key]))
        return masked_factors

    def get_unsqueezed_factor_masks(self, input_mask_dict=None):
        masked_factors = []
        for f in self.factors:
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, input_mask_dict)
            mask = input_mask_dict[mask_key]
            masked_factors.append(mask.unsqueeze(-1).expand(-1, -1, f.size))
        return masked_factors

    def masked_model_input(self, mask_dict):
        """Masks the pre-computed model input"""
        # Will return a [num_trajs, seq_len, factor_1_size + ... + factor_n_size] tensor
        masks = torch.cat(self.get_unsqueezed_factor_masks(mask_dict), dim=2)
        assert masks.shape == self.input.shape
        out = FactorSeq.mask_data(self.input, masks)
        desired_shape = (self.num_seqs, self.seq_len, self.factor_sizes.sum())
        assert out.shape == desired_shape, "{} vs {}".format(out.shape, desired_shape)
        return out

    def model_input(self):
        factors = [f.input for f in self.factors if f.name != "timestep"]
        return torch.cat(factors, dim=2)

    def __getitem__(self, key):
        return self.__class__(self.name, [f.__getitem__(key) for f in self.factors])

    def get_loss(self, loss_weights, pred_mask_dict):
        loss_d = {}
        for f in self.factors:
            # When computing losses, don't compute it for the timestep factor (if present) as we won't be predicting it
            if f.name == "timestep":
                continue
            mask_key = self.get_mask_key(f.name, pred_mask_dict)
            loss_d[f.name] = f.get_loss(loss_weights[f.name], pred_mask_dict[mask_key])
        return loss_d

    def get_factor(self, factor_name):
        for f in self.factors:
            if f.name == factor_name:
                return f
        raise ValueError("Factor not found")


def get_masked_items(items, mask):
    assert items.shape[:-1] == mask.shape[:-1], (items.shape, mask.shape)
    masked_indices = mask.reshape(-1).nonzero().ravel()
    if len(items.shape) == 1:
        masked_items = items[
            masked_indices,
        ]
    elif len(items.shape) == 2:
        masked_items = items.reshape(-1)[
            masked_indices,
        ]
    elif len(items.shape) == 3:
        n1, n2 = items.shape[:2]
        masked_items = items.reshape(n1 * n2, -1)[
            masked_indices,
        ]
    else:
        raise ValueError("{}".format(items.shape))
    return masked_items.to(device=device)


def get_accuracy(output_masked, targets_masked):
    """Expecting output_masked to be probabilities (or logits, as argmaxing leads to same result)"""
    return (output_masked.argmax(axis=1).squeeze() == targets_masked.squeeze()).float().mean()


def dict_equal_check(self, other, keys_to_ignore):
    for k in self.__dict__.keys():
        if k in keys_to_ignore:
            continue
        v0, v1 = self.__dict__[k], other.__dict__[k]
        if isinstance(v0, torch.Tensor):
            curr_v_equal = (v0 == v1).all().item()
        else:
            curr_v_equal = v0 == v1

        if not curr_v_equal:
            return False
    return True
