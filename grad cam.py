import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# 加载模型
model = keras.models.load_model('COVID densenet121.h5')

# 设定最后的卷积层名称
last_conv_layer_name = "relu"  # 请替换成你模型中最后一个卷积层的名字


# Grad-CAM函数
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # 首先，我们创建一个模型，它在给定模型的输入下，输出最后的卷积层和原始模型的输出
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # 然后，我们计算类别预测的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 这是输出特征图的梯度
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # 每个特征图的导数平均值，这是Grad-CAM的权重
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    # 我们将权重应用到最后一个卷积层的输出上
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # 为了使热图可视化，我们先将其标准化
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def preprocess_image(img_path):
    # 使用和训练模型时相同的预处理步骤
    img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # 确保和训练时的预处理一致
    return img


def display_heatmap(image_path, heatmap):
    img = cv2.imread(image_path)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap onto the original image
    superimposed_img = heatmap * 0.4 + img * 0.6  # Weighted sum
    superimposed_img = np.clip(superimposed_img, 0, 255)  # Ensure the values are within [0, 255]

    # Convert the superimposed image to uint8
    superimposed_img = np.uint8(superimposed_img)

    # Convert BGR to RGB for displaying
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # Display both the original image and the heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.axis('off')  # 关闭坐标轴
    plt.imshow(superimposed_img)
    #plt.title("Grad-CAM Heatmap")
    plt.show()

def preprocess_image(img_path, target_size=(224, 224)):
    # 使用和训练模型时相同的预处理步骤
    img = keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # 确保和训练时的预处理一致
    return img
# 主函数
def main(img_path):
    img_array = preprocess_image(img_path)

    # 预测图像
    preds = model.predict(img_array)
    pred_index = tf.argmax(preds[0])
    print(f"Predicted class: {labels[pred_index]} with confidence {preds[0][pred_index]:.2f}")

    # 生成Grad-CAM热图
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)

    # 显示Grad-CAM热图
    display_heatmap(img_path, heatmap)

# 文件夹路径
folder_path = "C:\\Users\\DELL\\Desktop\\code\\dataset\\COVID-19\\new_train\\2"

# 读取文件夹中的所有图像并为每个图像生成热度图
for img_file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, img_file)
    if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # 检查文件是否为图像
        main(img_path)