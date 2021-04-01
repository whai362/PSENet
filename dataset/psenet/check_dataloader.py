from psenet_ctw import PSENET_CTW
import torch
import numpy as np
import cv2
import random
import os

torch.manual_seed(123456)
torch.cuda.manual_seed(123456)
np.random.seed(123456)
random.seed(123456)


def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1], 1)
    img = np.concatenate((img, img, img), axis=2) * 255
    return img


def save(img_path, imgs):
    if not os.path.exists('vis/'):
        os.makedirs('vis/')

    for i in range(len(imgs)):
        imgs[i] = cv2.copyMakeBorder(imgs[i], 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255, 0, 0])
    res = np.concatenate(imgs, axis=1)
    if type(img_path) != str:
        img_name = img_path[0].split('/')[-1]
    else:
        img_name = img_path.split('/')[-1]
    print('saved %s.' % img_name)
    cv2.imwrite('vis/' + img_name, res)



# data_loader = SynthLoader(split='train', is_transform=True, img_size=640, kernel_scale=0.5, short_size=640,
#                           for_rec=True)
# data_loader = IC15Loader(split='train', is_transform=True, img_size=736, kernel_scale=0.5, short_size=736,
#                          for_rec=True)
# data_loader = CombineLoader(split='train', is_transform=True, img_size=736, kernel_scale=0.5, short_size=736,
#                             for_rec=True)
# data_loader = TTLoader(split='train', is_transform=True, img_size=640, kernel_scale=0.8, short_size=640,
#                        for_rec=True, read_type='pil')
# data_loader = CombineAllLoader(split='train', is_transform=True, img_size=736, kernel_scale=0.5, short_size=736,
#                                for_rec=True)
data_loader = PSENET_CTW(split='test', is_transform=True, img_size=736)
# data_loader = MSRALoader(split='train', is_transform=True, img_size=736, kernel_scale=0.5, short_size=736,
#                          for_rec=True)
# data_loader = CTWv2Loader(split='train', is_transform=True, img_size=640, kernel_scale=0.7, short_size=640,
#                           for_rec=True)
# data_loader = IC15(split='train', is_transform=True, img_size=640,)

train_loader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    drop_last=True)

for batch_idx, imgs in enumerate(train_loader):
    if batch_idx > 100:
        break
    # image_name = data_loader.img_paths[batch_idx].split('/')[-1].split('.')[0]

    # print('%d/%d %s'%(batch_idx, len(train_loader), data_loader.img_paths[batch_idx]))
    print('%d/%d' % (batch_idx, len(train_loader)))

    img = imgs[0].numpy()
    img = ((img * np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1) +
            np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)) * 255).astype(np.uint8)
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].copy()

    # gt_text = to_rgb(gt_texts[0].numpy())
    # gt_kernel_0 = to_rgb(gt_kernels[0, 0].numpy())
    # gt_kernel_1 = to_rgb(gt_kernels[0, 1].numpy())
    # gt_kernel_2 = to_rgb(gt_kernels[0, 2].numpy())
    # gt_kernel_3 = to_rgb(gt_kernels[0, 3].numpy())
    # gt_kernel_4 = to_rgb(gt_kernels[0, 4].numpy())
    # gt_kernel_5 = to_rgb(gt_kernels[0, 5].numpy())
    # gt_text_mask = to_rgb(training_masks[0].numpy().astype(np.uint8))


    # save('%d.png' % batch_idx, [img, gt_text, gt_kernel_0, gt_kernel_1, gt_kernel_2, gt_kernel_3, gt_kernel_4, gt_kernel_5, gt_text_mask])
    save('%d_test.png' % batch_idx, [img])