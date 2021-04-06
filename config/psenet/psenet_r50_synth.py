model = dict(
    type='PSENet',
    backbone=dict(
        type='resnet50',
        pretrained=True
    ),
    neck=dict(
        type='FPN',
        in_channels=(256, 512, 1024, 2048),
        out_channels=128
    ),
    detection_head=dict(
        type='PSENet_Head',
        in_channels=1024,
        hidden_dim=256,
        num_classes=7,
        loss_text=dict(
            type='DiceLoss',
            loss_weight=0.7
        ),
        loss_kernel=dict(
            type='DiceLoss',
            loss_weight=0.3
        )
    )
)
data = dict(
    batch_size=16,
    train=dict(
        type='PSENET_Synth',
        is_transform=True,
        img_size=736,
        short_size=736,
        kernel_num=7,
        min_scale=0.7,
        read_type='cv2'
    )
)
train_cfg = dict(
    lr=1e-3,
    schedule=(200, 400,),
    epoch=1,
    optimizer='SGD'
)

