"""sc_template: a small, fast LIBERO policy to copy when trying a new idea (lerobot out-of-tree plugin)."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError("lerobot is not installed. Install lerobot to use this policy package.") from e

from .configuration_sc_template import ScTemplateConfig
from .modeling_sc_template import ScTemplatePolicy
from .processor_sc_template import make_sc_template_pre_post_processors

__all__ = ["ScTemplateConfig", "ScTemplatePolicy", "make_sc_template_pre_post_processors"]
