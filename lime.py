import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import matplotlib.pyplot as plt

# 加载模型和图像
loaded_model = load_model('COVID densenet121.h5')
img_path ="C:\\Users\\DELL\\Desktop\\code\\dataset\\COVID-19\\test\\1\\COVIDCTMD-cap004-IM0070.png"
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 模型预测和获取类别索引
predicted_classes = loaded_model.predict(img_array)
class_labels = ['0', '1', '2', ]  # 按照训练时的顺序
predicted_label = class_labels[np.argmax(predicted_classes)]
#print(f"Model predicts: {predicted_label}")

# 确定‘meningioma_tumor’的索引
tumor_label_index = class_labels.index('1') if '1' in class_labels else None
if tumor_label_index is not None:
    print(f"Index of '0': {tumor_label_index}")
else:
    raise ValueError("'0' is not in the class labels.")

# 定义LIME解释器环境
class LimeEnv:
    def __init__(self, model, img_array):
        self.model = model
        self.img_array = img_array

    def get_lime_explanation(self):
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(self.img_array[0], self.model.predict,
                                                 top_labels=4, hide_color=0, num_samples=100,
                                                 segmentation_fn=SegmentationAlgorithm('felzenszwalb', scale=150,sigma=0.5, min_size=30))
        return explanation

env = LimeEnv(loaded_model, img_array)
explanation = env.get_lime_explanation()

env = LimeEnv(loaded_model, img_array)
explanation = env.get_lime_explanation()

# 可视化解释
fig, ax1 = plt.subplots(1, 1)
ax1.imshow(img_array[0])
ax1.axis('off')
temp, mask = explanation.get_image_and_mask(tumor_label_index, positive_only=True, num_features=10, hide_rest=True)
ax1.imshow(mask, cmap='coolwarm', alpha=0.5)

# 保存解释图像
image_save_path = "C:\\Users\\DELL\\Desktop\\segmentation\\m (3).jpg"
plt.imshow(img_array[0])
plt.imshow(mask, cmap='coolwarm', alpha=0.5)
plt.axis('off')
plt.savefig(image_save_path, bbox_inches='tight', pad_inches=0)
plt.show()
print(f"Image saved to {image_save_path}")
