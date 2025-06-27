"""
Frame samplers.
"""

import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple, Union


class FrameSamplerOutput(NamedTuple):
    """
    Return indices for frame decoding,
    and optionally additional information to return to user.
    """

    indices: List[int]
    additional_info: Dict[str, Any] = {}


class FrameSampler(ABC):
    """
    Frame sampler base class.

    Child class must implement __call__ method to return the decoding indices.
    Or raise if the video cannot be sampled (e.g. too short, etc.)
    """

    @abstractmethod
    def __call__(self, num_frames: int) -> FrameSamplerOutput:
        raise NotImplementedError


class AllFrameSampler(FrameSampler):
    """
    All frame sampler. Returns all frames in a video.
    """

    def __call__(self, num_frames: int) -> FrameSamplerOutput:
        return FrameSamplerOutput(list(range(num_frames)))

class FirstFrameSampler(FrameSampler):
    """
    All frame sampler. Returns all frames in a video.
    """

    def __call__(self, num_frames: int, frames_meta=None) -> FrameSamplerOutput:
        return FrameSamplerOutput([0])
class AdaptiveFrameSampler(FrameSampler):
    """
    Adaptive frame sampler.

    Arguments:
        length: frame length to return.
                For example, [5,10] denotes to always return 5 frames or 10 frames.
                It will choose the longest length that fits the original video.
                For example, if the video is 9 frames total, it will clip to 5 frames.
        stride: frame skip.
                For example, 1 denotes no skip. 2 denotes select every other frame. 3
                denotes select every third frame. When a list is given, stride is randomly
                chosen with even probability. However, user may set it to [1,1,2] to
                denote 1 with 66% probability and 2 with 33% proability.
        clip:   clip location.
                    "center":   clip video at the center.
                    "uniform":  clip video uniformly at random.
        jitter: jitter to the location.
                Only applicable when clip is "center".
                The value is the stdev of the normal distribution to shift the index.
    """

    def __init__(
        self,
        lengths: Union[int, List[int]],
        strides: Union[int, List[int]] = 1,
        clip: Literal["center", "uniform"] = "uniform",
        jitter: float = 0.0,
    ):
        lengths = [lengths] if isinstance(lengths, int) else lengths
        strides = [strides] if isinstance(strides, int) else strides
        assert len(lengths) > 0
        assert len(strides) > 0
        assert clip in ["center", "uniform"]
        assert jitter >= 0
        self.lengths = np.array(lengths)
        self.strides = np.array(strides)
        self.clip = clip
        self.jitter = jitter

    def __call__(
        self,
        num_frames: int,
    ) -> FrameSamplerOutput:
        # Choose stride.
        # Drop strides that are too long for this video.
        # Then randomly choose a valid stride.
        valid_strides = np.any(num_frames // self.strides >=
                               self.lengths.reshape(-1, 1), axis=0)
        valid_strides = self.strides[valid_strides]
        if valid_strides.size <= 0:
            raise ValueError(f"Video is too short ({num_frames} frames).")
        stride = np.random.choice(valid_strides)

        # Choose length.
        # Pick the max length that can fit the video under the current stride.
        valid_lengths = self.lengths[num_frames // stride >= self.lengths]
        length = np.max(valid_lengths)

        # Choose start index.
        min_start_index = 0
        max_start_index = num_frames - 1 - stride * (length - 1)
        mid_start_index = round((min_start_index + max_start_index) / 2)
        jitter = round(np.random.normal(loc=0, scale=self.jitter))

        if self.clip == "center":
            start_index = mid_start_index + jitter
        elif self.clip == "uniform":
            start_index = np.random.randint(
                min_start_index, max_start_index + 1)
        else:
            raise NotImplementedError

        start_index = np.clip(start_index, min_start_index, max_start_index)

        # Compute indices
        indices = np.arange(start_index, start_index + length * stride, stride)

        # Return indices and additional information to return to user.
        return FrameSamplerOutput(
            indices=indices.tolist(),
            additional_info={
                "stride": stride,
                "start_frame": start_index,
                "end_frame": start_index + length * stride,
                "total_frames": num_frames,
            },
        )


@dataclass
class AdaptiveAdvancedFrameSamplerStrategy:
    stride: int
    stride_prob: float
    frame_lengths: List[int]
    frame_lengths_prob: Union[Literal["uniform", "harmonic"], List[float]]


class AdaptiveAdvancedFrameSampler(FrameSampler):
    """
    Advanced adaptive frame sampler supports different frame lengths for different strides,
    and supports probabilistic sampling of both the stride and the frame length.

    strategies: A list of strategies to sample from.
    clip:   clip location.
            "center":   clip video at the center.
            "uniform":  clip video uniformly at random.
    jitter: jitter to the location.
            Only applicable when clip is "center".
            The value is the stdev of the normal distribution to shift the index.
    """

    def __init__(
        self,
        strategies: List[AdaptiveAdvancedFrameSamplerStrategy],
        clip: Literal["center", "uniform","simple"] = "uniform",
        jitter: float = 0.0,
        aligned: bool = False,
    ):
        assert len(strategies) > 0, "Strategies must not be empty"
        assert len({s.stride for s in strategies}) == len(
            strategies), "Strides cannot duplicate."
        assert clip in ["center", "uniform","simple"]
        assert jitter >= 0
        self.aligned = aligned
        self.clip = clip
        self.jitter = jitter
        self.strides = []
        self.strides_prob = []
        self.frame_lengths = []
        self.frame_lengths_prob = []

        for strategy in sorted(strategies, key=lambda s: s.stride):
            # Validate strides.
            assert isinstance(
                strategy.stride, int), "Stride must be an integer."
            assert strategy.stride > 0, "Stride must be a positive integer."
            self.strides.append(strategy.stride)

            # Assign strides_prob.
            assert isinstance(strategy.stride_prob, (int, float)
                              ), "Stride prob is not int/float."
            assert strategy.stride_prob >= 0, "Stride prob must be non-negative."
            self.strides_prob.append(strategy.stride_prob)

            # Assign frame lengths, sort by value.
            assert len(
                strategy.frame_lengths) > 0, "Frame lengths must not be empty."
            frame_lengths = np.array(strategy.frame_lengths)
            assert frame_lengths.dtype == int, "Frame lengths must be integers."
            assert np.all(frame_lengths >
                          0), "Frame lengths must be positive integers."
            frame_lengths_sorted_idx = np.argsort(frame_lengths)
            frame_lengths = frame_lengths[frame_lengths_sorted_idx]
            self.frame_lengths.append(frame_lengths)

            # Assign frame lengths prob, apply the sorting to prob as well.
            if strategy.frame_lengths_prob == "uniform":
                # e.g. [0.2, 0.2, 0.2, 0.2, 0.2]
                frame_lengths_prob = np.full(
                    len(frame_lengths), 1.0 / len(frame_lengths))
            elif strategy.frame_lengths_prob == "harmonic":
                # e.g. [0.2, 0.25, 0.33, 0.5, 1]
                frame_lengths_prob = np.flip(
                    1 / np.arange(1, len(frame_lengths) + 1))
            elif isinstance(strategy.frame_lengths_prob, list):
                frame_lengths_prob = np.array(strategy.frame_lengths_prob)
                frame_lengths_prob = frame_lengths_prob[frame_lengths_sorted_idx]
            else:
                raise NotImplementedError
            assert len(frame_lengths_prob) == len(
                frame_lengths), "Frame lengths prob mismatch."
            assert np.all(frame_lengths_prob >=
                          0), "Frame lengths prob must not be negative."
            assert frame_lengths_prob.sum() > 0, "Frame lengths prob must not be all zeros."
            frame_lengths_prob /= frame_lengths_prob.sum()
            self.frame_lengths_prob.append(frame_lengths_prob)

        self.strides = np.array(self.strides)
        self.strides_prob = np.array(self.strides_prob)
        assert self.strides_prob.sum() > 0, "Strides prob must not be all zeros."
        self.strides_prob /= self.strides_prob.sum()

    def __call__(self, num_frames: int, frames_meta=None):
        global_start_idx, global_end_idx = 0, num_frames
        if self.aligned:
            assert frames_meta is not None
            global_start_idx = frames_meta['start_idxs']
            global_end_idx = frames_meta['end_idxs']
        num_frames = global_end_idx - global_start_idx

        if self.clip != 'simple':
            sample_result = adptive_sample_framelen_and_stride(
                num_frames=num_frames,
                strides=self.strides,
                strides_prob=self.strides_prob,
                frame_lengths=self.frame_lengths,
                frame_lengths_prob=self.frame_lengths_prob,
            )

            stride = sample_result["stride"]
            length = sample_result["frame_length"]
        else:
            stride = self.strides[0]
            length = self.frame_lengths[0][0]

        # Choose start index.
        min_start_index = 0
        max_start_index = num_frames - 1 - stride * (length - 1)
        mid_start_index = round((min_start_index + max_start_index) / 2)
        jitter = round(np.random.normal(loc=0, scale=self.jitter))

        if self.clip == 'simple':
            start_index = global_start_idx
        ## can only load dump data, will fix further
        # if self.clip == "center":
        #     start_index = mid_start_index + jitter
        elif self.clip == "uniform":
            start_index = np.random.randint(
                min_start_index, max_start_index + 1)
        # else:
        #     raise NotImplementedError
        # else:
        #     start_index += global_start_idx
        #     min_start_index += global_start_idx
        #     max_start_index += global_start_idx
        #     start_index = np.clip(start_index, min_start_index, max_start_index)

        # Compute indices
        indices = np.arange(start_index, start_index + length * stride, stride)

        # Return indices and additional information to return to user.
        return FrameSamplerOutput(
            indices=indices.tolist(),
            additional_info={
                "stride": stride,
                "start_frame": start_index,
                "end_frame": start_index + length * stride,
                "total_frames": num_frames,
            },
        )


def normalize_probabilities(
    items: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(items), "Items must not be empty."
    assert len(items) == len(masks) == len(probs), "Lengths must match."
    assert isinstance(items, np.ndarray), "Items must be an np.ndarray."
    assert isinstance(probs, np.ndarray), "Probs must be an np.ndarray."
    assert isinstance(masks, np.ndarray), "Masks must be an np.ndarray."
    assert masks.dtype == bool, "Masks must be boolean."
    assert np.any(masks), "Masks must not be all False."
    assert np.all(np.diff(masks.astype("int")) <=
                  0), "Masks must not break monotonicity."

    ret_items = items[masks]
    ret_probs = probs[masks]

    # Accumulate the probabilities of infeasible items to the last feasible one.
    ret_probs[-1] += probs[~masks].sum()

    return ret_items, ret_probs


def adptive_sample_framelen_and_stride(
    num_frames: int,
    strides: np.ndarray,
    strides_prob: np.ndarray,
    frame_lengths: List[np.ndarray],
    frame_lengths_prob: List[Optional[np.ndarray]],
) -> Dict[str, Any]:
    """Adaptively sample frame length and stride for a video.

    Args:
        num_frames: Number of frames in the current video.
        strides: A list of strides.
        strides_prob: The probability for each stride.
        frame_lengths: The number of frames (sorted) to sample from at the current stride.
            For example, `frame_length=10` at `stride=2` means that we need to have 20 frames.
            When the number of frames to sample is infeasible, it will select the feasible frame
            lengths and re-normalize the probability according to the feasible frames at hand.
            For example, if `num_frames=10`, `frame_lengths[stride2]=[4, 5]`,
            `frame_lengths[stride3]=[1, 3, 5]`, we can sample frame lengths 1, 2, and 5 at
            `stride=2` (2, 4, and 10 frames) but only frame lengths 1, 3 at `stride=3`. In this
            case, we will add the probability of `frame_length=5` at `stride=3` to `frame_length=3`
            at `stride=3`, making it more likely to be selected.
        frame_lengths_prob: The frame probabilities to sample from the corresponding frame lengths.
            Defaults to None for uniform sampling.
    Returns:
        dictionary: A dictionary containing the selected frames and strides. if none is feasible,
        it will raise an exception.
    """
    assert len(strides) == len(strides_prob) == len(
        frame_lengths) == len(frame_lengths_prob)

    # Prepare frame_lengths_mask for each stride.
    frame_lengths_mask = [num_frames // s >=
                          l for s, l in zip(strides, frame_lengths)]

    # Prepare stride mask and prob.
    strides_idxs = np.arange(len(strides))
    strides_mask = np.array([np.any(mask) for mask in frame_lengths_mask])
    assert np.any(strides_mask), (
        f"Cannot sample frames={num_frames} "
        + f"from strides={strides} and lengths={frame_lengths}"
    )

    # Drop infeasible strides and normalize probability.
    strides_idxs, strides_prob = normalize_probabilities(
        strides_idxs, strides_prob, strides_mask)

    # Choose stride.
    stride_idx = np.random.choice(strides_idxs, p=strides_prob)
    stride = strides[stride_idx]

    # Prepare frame_lengths mask and prob for the current stride.
    lengths = frame_lengths[stride_idx]
    lengths_mask = frame_lengths_mask[stride_idx]
    lengths_prob = frame_lengths_prob[stride_idx]
    if lengths_prob is None:
        lengths_prob = np.full(len(lengths), 1.0 / len(lengths))

    # Drop infeasible lengths and normalize probability.
    lengths, lengths_prob = normalize_probabilities(
        lengths, lengths_prob, lengths_mask)

    # Choose frame length.
    length = np.random.choice(lengths, p=lengths_prob)
    return dict(stride=stride, frame_length=length)
