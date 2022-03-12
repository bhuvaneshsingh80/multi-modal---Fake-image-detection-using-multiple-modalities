import cv2
import os
import pickle as pkl
from tensorflow.keras.models import Model
from tensorflow.keras.applications.efficientnet import EfficientNetB3, preprocess_input

model = EfficientNetB3(include_top=True)
new_model = Model(inputs=model.input, outputs=model.layers[-3].output)

def save_image_embeddings(phase):
    embeddings = dict()
    data_dir = './mediaeval2016/images_' + phase
    images_list = os.listdir(data_dir)
    filtered_images = [img for img in images_list if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    for image_name in filtered_images:
        image = cv2.imread(os.path.join(data_dir, image_name))
        image = cv2.resize(image, (300, 300))
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        image_embedding = new_model.predict(image)
        embeddings[image_name] = image_embedding

    pkl.dump(embeddings, open('./temp/' + phase + '_image_embeddings.pkl', 'wb'))
    print(phase + " image embeddings dumped with length: " + str(len(embeddings)))
    return


if __name__ == '__main__':
    save_image_embeddings('train')
    save_image_embeddings('test')
