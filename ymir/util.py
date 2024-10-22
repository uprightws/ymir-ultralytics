from ymir_exc.util import  get_weight_files
import os.path as osp
def get_weight_file(ymir_cfg) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    weight_files = get_weight_files(ymir_cfg, suffix=('.pt'))
    # choose weight file by priority, best.pt > xxx.pt
    for p in weight_files:
        if p.endswith('best.pt'):
            return p

    if len(weight_files) > 0:
        return max(weight_files, key=osp.getctime)

    return ""