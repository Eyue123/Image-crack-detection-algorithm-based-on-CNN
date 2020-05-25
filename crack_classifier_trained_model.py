import numpy as np
import cv2
import keras
import os
import math

import pca
import img_preprocess
import crop_and_compose

IMAGE_RESIZE = 320
PI = 3.14159
IMAGES_FORMAT = ['.jpg', '.JPG']

img_size = 32
transerse_crack_num = 0      # 横向裂纹数量
longitudinal_crack_num = 0   # 纵向裂纹数量
alligatot_crack_num = 0      # 鳄鱼裂纹数量
not_crack_num = 0            # 没有裂纹数量

preprocess_path = "/home/eyue/graduation_desigh_QinJiang/用于验证的图像/横向/preprocessed/"


def prepare_image(cropped_path, i ,j):
    file = os.path.join(cropped_path, "%s-%s.jpg" %(i, j))
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, (img_size, img_size))
    return img.reshape(-1, img_size, img_size, 1)


def become_dark(path):
    to_black_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    for x in range(to_black_img.shape[0]):
        for y in range(to_black_img.shape[1]):
            to_black_img[x, y] = 0
    cv2.imwrite(path, to_black_img)


print("Loading trained model...")
model = keras.models.load_model("/home/eyue/graduation_desigh_QinJiang/my_model/"
                                "Concrete_Crack_Classification_model.model")
print("Trained model loaded!")

preprocessd_save_path = img_preprocess.preprocess(preprocess_path, IMAGE_RESIZE)

image_names = [name for name in os.listdir(preprocessd_save_path) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]
# print(os.path.splitext(image_names[0])[0])

for m in range(0, len(image_names)):
    every_save_path = os.path.join(preprocessd_save_path, "%s" % os.path.splitext(image_names[m])[0])  # 存储每张图片分割后、重组后、以及评价的目录
    os.mkdir(every_save_path)
    preprocessd_img = os.path.join(preprocessd_save_path, image_names[m])
    print(preprocessd_img)
    save_to_every = cv2.imread(preprocessd_img, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(every_save_path, image_names[m]), save_to_every)


    cropped_path = crop_and_compose.image_crop(preprocessd_img, every_save_path)

    # 构造裂纹区域矩阵 n*2维
    area_array = np.empty(shape=[0, 2])

    # 创建并以 追加 方式打开评估文件
    evaluate_path = os.path.join(every_save_path, "evaluate.txt")
    os.mknod(evaluate_path)
    evaluate_txt = open(evaluate_path, "a")
    # 一张图中包含裂纹的区域数目
    crack_area_num = 0

    for i in range(0, 10):
        for j in range(0, 10):
            img = prepare_image(cropped_path, i, j)
            prediction = model.predict(img)
            if prediction[0][0] > .5:
                area_array = np.append(area_array, [[j+0.5, 9.5-i]], axis=0)
                crack_area_num += 1
            elif prediction[0][0] <= .5:
                path = os.path.join(cropped_path, "%s-%s.jpg" %(i, j))
                become_dark(path)
            else:
                print("\nSomething went wrong...")

    # 裂纹骨架图
    crop_and_compose.image_compose(cropped_path, every_save_path)

    if crack_area_num >= 2:
        # PCA分析
        pcs, s = pca.compute_pca(area_array)
        eigenvector = pcs[[0, 1]]
        eigenvalue = np.empty(shape=[0, 1])
        eigenvalue = np.append(eigenvalue, s[0])
        eigenvalue = np.append(eigenvalue, s[1])
        # print(eigenvector)
        # print(eigenvalue)

        # 计算线性相关系数
        LLC = max(eigenvalue)/min(eigenvalue)
        # print(LLC)

        evaluate_txt.write("特征向量：\n%s\n" % eigenvector)
        evaluate_txt.write("特征值：\n%s\n" % eigenvalue)
        evaluate_txt.write("线性相关系数 LLC ：\n%s\n" % LLC)

        if LLC > 1.8:
            index = np.argsort(eigenvalue)
            max_index = index[-1]   # 最大特征值的索引
            #print(max_index)
            ratio = abs(eigenvector[1, max_index])/abs(eigenvector[0, max_index])  # y与x的比值
            #print(ratio)
            angle = (180/PI) * math.atan(ratio)
            evaluate_txt.write("角度：\n%s\n" % angle)
            #print(angle)

            if (angle > 0 and angle < 45):
                print("是横向裂纹")
                transerse_crack_num += 1
                evaluate_txt.write("是横向裂纹")
            else:
                print("是纵向裂纹")
                longitudinal_crack_num += 1
                evaluate_txt.write("是纵向裂纹")
        else:
            print("是鳄鱼裂纹")
            alligatot_crack_num += 1
            evaluate_txt.write("是鳄鱼裂纹")

    else:
        print("没有裂纹")
        not_crack_num += 1
        evaluate_txt.write("没有裂纹")

    evaluate_txt.close()

print("横向裂纹的图像数量：%s\n" % transerse_crack_num)
print("纵向裂纹的图像数量：%s\n" % longitudinal_crack_num)
print("鳄鱼裂纹的图像数量：%s\n" % alligatot_crack_num)
print("没有裂纹的图像数量:%s\n" % not_crack_num)