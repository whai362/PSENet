import models


def build_loss(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    loss = models.loss.__dict__[cfg.type](**param)

    return loss
