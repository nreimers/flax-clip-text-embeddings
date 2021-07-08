import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class HybridCLIPConfig(PretrainedConfig):

    model_type = "hybrid-clip"
    is_composition = True

    def __init__(self, projection_dim=512, **kwargs):
        super().__init__(**kwargs)

        if "text_config" not in kwargs:
            raise ValueError("`text_config` can not be `None`.")



        text_config = kwargs.pop("text_config")
        text_model_type = text_config.pop("model_type")


        from transformers import AutoConfig

        self.text_config = AutoConfig.for_model(text_model_type, **text_config)
        self.projection_dim = projection_dim
        self.initializer_factor = 1.0

    @classmethod
    def from_text_configs(cls, text_config: PretrainedConfig, **kwargs):
        r"""
        Instantiate a :class:`HybridCLIPConfig` (or a derived class) from text model configuration and
        vision model configuration.

        Returns:
            :class:`HybridCLIPConfig`: An instance of a configuration object
        """

        return cls(text_config=text_config.to_dict(), **kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default
        :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
