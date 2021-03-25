import models


def build_head(cfg):
    param = dict()
    for key in cfg:
        if key == 'type':
            continue
        param[key] = cfg[key]

    head = models.head.__dict__[cfg.type](**param)

    return head
