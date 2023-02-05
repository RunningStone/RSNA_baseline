import numpy as np
from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip,
                            Compose, Resize, RandomBrightnessContrast,
                            HueSaturationValue, Blur, GaussNoise, Rotate,
                            RandomResizedCrop, Cutout, ShiftScaleRotate,
                            ToGray,OneOf,IAAAdditiveGaussianNoise,GaussNoise,
                            MotionBlur,MedianBlur)
from albumentations.pytorch import ToTensorV2


def get_transform(is_train = True):
    if is_train:
        transform = Compose([
            Resize(height=512, width=512, always_apply=True),
            # RandomResizedCrop(height=224, width=224),         # 可能导致缺失关键部位
            # Resize(height=512, width=512, always_apply=True), # 可能导致缺失关键部位
            
            OneOf([
                IAAAdditiveGaussianNoise(),   # 将高斯噪声添加到输入图像
                GaussNoise(),    # 将高斯噪声应用于输入图像。
            ], p=0.2),   # 应用选定变换的概率
            OneOf([
                MotionBlur(p=0.2),   # 使用随机大小的内核将运动模糊应用于输入图像。
                MedianBlur(blur_limit=3, p=0.1),    # 中值滤波
                Blur(blur_limit=3, p=0.1),   # 使用随机大小的内核模糊输入图像。
            ], p=0.2),

            ShiftScaleRotate(
                shift_limit = 0.05,
                rotate_limit = 30, 
                scale_limit = [-0.2, 0.2]
            ),

            # HorizontalFlip(p=horizontal_flip),     # 这样df里面的视角可能就会有问题
            # VerticalFlip(p=vertical_flip),         # 这样df里面的视角可能就会有问题

            RandomBrightnessContrast(p=0.2),   # 随机明亮对比度
            ToTensorV2()
        ])
    else:
        transform = Compose([
            Resize(height=512, width=512, always_apply=True),
            ToTensorV2()
        ])
    return transform


def get_three_channels(img):
    return np.concatenate([img, img, img], axis=0)


'''
1. HorizontalFlip 水平翻转 参数:
    p 翻转概率

2. ShiftScaleRotate 随机放射变换  该方法可以对图片进行平移(translate)、缩放(scale)和旋转(roatate),其含有以下参数:
    shift_limit: 图片宽高的平移因子,可以是一个值(float),也可以是一个元组((float, float))。
                 如果是单个值，那么可选值得范围是 [0,1]，之后该值会转换成（-shift_limit, shift_limit)。默认值为 (-0.0625, 0.0625)
    scale_limit: 图片缩放因子,可以是一个值(float),也可以是一个元组((float, float))。
                 如果是单个值，之后该值会转换成（-scale_limit, scale_limit)。默认值为 (-0.1, 0.1)
    rotate_limit: 图片旋转范围,可以是一个值(int),也可以是一个元组((int, int))。
                  如果是单个值，那么会被转换为 (-rotate_limit, rotate_limit)。默认值为(-45, 45)
    interpolation: OpenCV 标志,用于指定使用的差值算法,这些差值算法必须是cv2.INTER_NEAREST, cv2.INTER_LINEAR, 
                   cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4 中的一个。默认是 cv2.INTER_LINEAR
    border_mode: OpenCV 标志,用于指定使用的外插算法(extrapolation),算法必须是cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE, 
                cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_REFLECT_101中的一个,默认为 cv2.BORDER_REFLECT_101
    value: 当 border_mode 的值为 cv2.BORDER_CONSTANT 时,进行填补的值,该值就可以时一个int或者float值,也可以是int或者 float数组
    mask_value: 当 border_mode 的值为 cv2.BORDER_CONSTANT 时，应用到 mask 的填充值。
    p: 使用此转换的概率，默认值为 0.5

3. Compose 组合变换 参数:
    transforms: 转换类的数组:list类型
    bbox_params: 用于 bounding boxes 转换的参数:BboxPoarams 类型
    keypoint_params: 用于 keypoints 转换的参数， KeypointParams 类型
    additional_targets: key新target 名字,value 为旧 target 名字的 dict,如 {'image2': 'image'},dict 类型
    p: 使用这些变换的概率，默认值为 1.0

4. OneOf 执行列表中操作的一个
    transforms:转换类的列表
    p:使转换方法的概率，默认值为 0.5




'''
