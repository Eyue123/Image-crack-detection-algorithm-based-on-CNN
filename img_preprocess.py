import cv2
import os

IMAGES_FORMAT = ['.jpg', '.JPG']

# preprocess_path = "/home/eyue/tf_keras_API/纵向/"


def preprocess(preprocess_path, image_resize):
    save_path = os.path.join(preprocess_path, "preprocessed_images/")
    os.mkdir(save_path)
    image_names = [name for name in os.listdir(preprocess_path) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]

    for i in range(0, len(image_names)):
        # print(image_names[i])
        path = os.path.join(preprocess_path, image_names[i])
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_resize, image_resize))
        img = cv2.bilateralFilter(img, 4, 10, 10)  # 双边滤波
        img = cv2.GaussianBlur(img, (5, 5), 1)  # 高斯滤波
        # cv2.imshow("after_preprocess", img)
        # cv2.waitKey()
        img_save_path = os.path.join(save_path, "%s.jpg" % i)
        cv2.imwrite(img_save_path, img)

    return save_path




