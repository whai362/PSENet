import models


def build_model(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    model = models.__dict__[cfg.type](**param)

    return model
