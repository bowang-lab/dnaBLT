# import torch
# from transformers import AutoConfig, AutoModelForCausalLM

# hf_model_name = "togethercomputer/evo-1-8k-base"
# model_config = AutoConfig.from_pretrained(
#     hf_model_name,
#     trust_remote_code=True,
#     revision='1.1_fix',
# )

# model = AutoModelForCausalLM.from_pretrained(
#     hf_model_name,
#     config=model_config,
#     trust_remote_code=True,
#     revision='1.1_fix',
# )

# torch.save(model.backbone.state_dict(), "evo7b_state_dict.pt")

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
import torch
import yaml

class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

with open('evo-1-8k-base_inference.yml', 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)

global_config = dotdict(data)
# state_dict = torch.load("evo7b_state_dict.pt")
model = StripedHyena(global_config)
# model.load_state_dict(state_dict, strict=True)
# model.to_bfloat16_except_poles_residues()
print(sum([p.numel() for p in model.parameters()]))