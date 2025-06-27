from omegaconf import DictConfig, OmegaConf

from .text_sampler import (TextAllSampler,
                           TextFrequencySampler,
                           TextPrioritySampler,
                           TextSampler,
                           )
from .frame_sampler import (AdaptiveAdvancedFrameSampler,
                            AdaptiveAdvancedFrameSamplerStrategy,
                            AdaptiveFrameSampler,
                            AllFrameSampler,
                            FrameSampler,
                            FirstFrameSampler
                            )

TEXT_SAMPLER_TYPES = {
    "all": TextAllSampler,
    "frequency": TextFrequencySampler,
    "priority": TextPrioritySampler,
}


def create_text_sampler(config: dict) -> TextSampler:
    config = OmegaConf.to_object(config)
    sampler_type = config.pop("type")
    return TEXT_SAMPLER_TYPES[sampler_type](**config)


FRAME_SAMPLER_TYPES = {
    "all": AllFrameSampler,
    "adaptive": AdaptiveFrameSampler,
    "adaptive_advanced": AdaptiveAdvancedFrameSampler,
    "first_frame":FirstFrameSampler
}

def create_frame_sampler(config: dict) -> FrameSampler:
    config = OmegaConf.to_object(config)
    sampler_type = config.pop("type")
    if sampler_type == "adaptive_advanced":
        config["strategies"] = [
            AdaptiveAdvancedFrameSamplerStrategy(**s) for s in config["strategies"]
        ]
    return FRAME_SAMPLER_TYPES[sampler_type](**config)
