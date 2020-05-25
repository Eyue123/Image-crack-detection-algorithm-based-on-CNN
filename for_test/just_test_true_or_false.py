import cv2
import keras
import os

IMAGES_FORMAT = ['.jpg', '.JPG']
img_size = 32
is_true_num = 0    # 包含裂纹的图像数目
is_false_num = 0   # 不包含裂纹的图像数目
img_path_to_predict = "/home/eyue/graduation_desigh_QinJiang/crack_images_of_testing_set/无裂纹/"


def prepare_image(file):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.bilateralFilter(img, 4, 10, 10)  # 双边滤波
    img = cv2.GaussianBlur(img, (5, 5), 0.6)  # 高斯滤波
    return img.reshape(-1, img_size, img_size, 1)


print("Loading trained model...")
model = keras.models.load_model("/home/eyue/graduation_desigh_QinJiang/my_model/"
                                "Concrete_Crack_Classification_model.model")
print("Trained model loaded!")

image_names = [name for name in os.listdir(img_path_to_predict) for item in IMAGES_FORMAT if
                   os.path.splitext(name)[1] == item]

for i in range(0, len(image_names)):
    img_to_predict = os.path.join(img_path_to_predict, image_names[i])
    #print("Model predicting...")
    prediction = model.predict([prepare_image(img_to_predict)])

    if prediction[0][0] <= .5:
        # pred_text = "Networks prediction:\nThis surface DOES NOT
        # have a crack on it. Confidence: {:.2f}%".format((1 - prediction[0][0]) * 100)
        is_false_num += 1
    elif prediction[0][0] > .5:
        # pred_text = "Networks prediction:\nThis surface DOES
        # have a crack on it. Confidence: {:.2f}%".format((prediction[0][0]) * 100)
        is_true_num += 1
    else:
        print("\nSomething went wrong...")

    # plt.imshow(cv2.resize(cv2.imread(img_to_predict), (img_size, img_size)))
    # plt.title("What the Neural Network is receiving as input:")
    # plt.text(2, 5, pred_text, fontweight = "bold")
    # plt.show()

print("有裂纹的图像数目：%s" % is_true_num)
print("没有裂纹的图像数目：%s" % is_false_num)
