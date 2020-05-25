import cv2
import os
from PIL import Image

IMAGE_ROW = 10  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 10  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGES_FORMAT = ['.jpg', '.JPG']
IMAGE_SIZE = 32


def image_crop(to_crop_image, every_save_path):
    cropped_path = os.path.join(every_save_path, "cropped/")
    os.mkdir(cropped_path)
    img = cv2.imread(to_crop_image, cv2.IMREAD_GRAYSCALE)

    for j in range(0, IMAGE_ROW):
        for i in range(0, IMAGE_COLUMN):
            img_cropped = img[i * 32:(i + 1) * 32, j * 32:(j + 1) * 32]
            save_path = os.path.join(cropped_path, "%d-%d.jpg" % (i, j))
            cv2.imwrite(save_path, img_cropped)

    return cropped_path


def image_compose(cropped_path, every_save_path):
    composed_save_path = os.path.join(every_save_path, "composed/")
    os.mkdir(composed_save_path)
    cropped_image_names = [name for name in os.listdir(cropped_path) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
    cropped_image_names.sort()
    if len(cropped_image_names) != IMAGE_ROW * IMAGE_COLUMN:
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    new_image = Image.new("L", (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE))  # 创建一个新图
    i = 0
    for y in range(0, IMAGE_COLUMN):
        for x in range(0, IMAGE_ROW):
            cropped_imag_path = os.path.join(cropped_path, cropped_image_names[i])
            cropped_imag = Image.open(cropped_imag_path, "r")
            new_image.paste(cropped_imag, (x * IMAGE_SIZE, y * IMAGE_SIZE))
            i += 1

    save_path = os.path.join(composed_save_path, "composed.jpg")
    new_image.save(save_path)  # 保存新图
    return composed_save_path
