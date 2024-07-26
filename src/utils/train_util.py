import importlib
import torch

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def get_central_point_of_patch(image_size, patch_size=16):
    height, width = image_size, image_size
    central_pixels = []

    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Find the central pixel of the patch
            central_i = i + patch_size / 2
            central_j = j + patch_size / 2

            central_pixels.append((central_i, central_j))
    
    return torch.tensor(central_pixels).reshape(1, (image_size//patch_size)**2, 2)
