import numpy as np
import pickle as pkl
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

model = SentenceTransformer("roberta-base-nli-stsb-mean-tokens")

def save_text_embeddings(phase):
    tweet_data = list()
    image_data = list()
    label_data = list()
    dataset = pkl.load(open('./temp/' + phase + '_dataset.pkl', 'rb'))
    image_embeddings = pkl.load(open('./temp/' + phase + '_image_embeddings.pkl', 'rb'))
    for data in tqdm(dataset):
        image_name = data[2]
        if image_name in image_embeddings:
            image_data.append(image_embeddings[image_name][0])
            tweet_embedding = model.encode(data[1])
            tweet_data.append(tweet_embedding)
            label_data.append(data[3])
    tweet_data = np.array(tweet_data)
    image_data = np.array(image_data)
    label_data = np.array(label_data)
    np.save('./temp/' + phase + '_text', tweet_data)
    np.save('./temp/' + phase + '_image', image_data)
    np.save('./temp/' + phase + '_label', label_data)
    print(f"Size of {phase} text input {tweet_data.size} and shape of {phase} text input {tweet_data.shape}")
    print(f"Size of {phase} image input {image_data.size} and shape of {phase} image input {image_data.shape}")
    print(f"Size of {phase} label input {label_data.size} and shape of {phase} label input {label_data.shape}")
    return


if __name__ == '__main__':
    save_text_embeddings('train')
    save_text_embeddings('test')
