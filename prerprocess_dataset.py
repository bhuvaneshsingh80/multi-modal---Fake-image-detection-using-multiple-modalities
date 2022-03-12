import os
import pickle as pkl
from googletrans import Translator
from langdetect import detect
from tokenizer import tokenize
from tqdm import tqdm
import re
from gensim.parsing.preprocessing import STOPWORDS

translator = Translator()

def tokenize(text):
    try:
        text = text.decode('utf-8').lower()
    except:
        text = text.encode('utf-8').decode('utf-8').lower()
    text = re.sub(u"\u2019|\u2018", "\'", text)
    text = re.sub(u"\u201c|\u201d", "\"", text)
    text = re.sub(u"[\u2000-\u206F]", " ", text)
    text = re.sub(u"[\u20A0-\u20CF]", " ", text)
    text = re.sub(u"[\u2100-\u214F]", " ", text)
    text = re.sub(r"http:\ ", "http:", text)
    text = re.sub(r"http[s]?:[^\ ]+", " ", text)
    text = re.sub(r"&gt;", " ", text)
    text = re.sub(r"&lt;", " ", text)
    text = re.sub(r"&quot;", " ", text)
    text = re.sub(r"\"", " ", text)
    text = re.sub(r"#\ ", "#", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"[\(\)\[\]\{\}]", r" ", text)
    text = re.sub(u'['
                  u'\U0001F300-\U0001F64F'
                  u'\U0001F680-\U0001F6FF'
                  u'\u2600-\u26FF\u2700-\u27BF]+',
                  r" ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " had ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"#", " #", text)
    text = re.sub(r"\@", " \@", text)
    text = re.sub(r"[\!\?\.\,\+\-\$\%\^\>\<\=\:\;\*\(\)\{\}\[\]\/\~\&\'\|]", " ", text)

    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    if len(words) < 3:
        return None
    return ' '.join(words)

def process_dataset(phase):
    f = open('C:/Users/singhbhu/Desktop/mediaeval2016/' + phase + '_posts.txt', encoding="utf8")
    images_list = os.listdir('C:/Users/singhbhu/Desktop/mediaeval2016/images_' + phase)
    filtered_images = [img for img in images_list if (img.endswith('.png') or img.endswith('.jpg') or img.endswith('.jpeg'))]
    filtered_images_without_format = [img.split(".")[0] for img in filtered_images]
    images_dict = dict(zip(filtered_images_without_format, filtered_images))
    tweets = f.readlines()
    tweets = tweets[1:]
    processed_tweet = []
    ct = 0
    for tweet in tqdm(tweets):
        original_tweet_text = tweet.split('\t')[1]
        tweet_text = tweet.split('\t')[1]
        #try:
            #if detect(tweet_text) != 'en':
        print(tweet_text)    
        tweet_text = translator.translate(tweet_text, dest='en').text
        ct += 1
        print(tweet_text)
        #except:
            #continue

        tweet_text = tokenize(tweet_text)

        if tweet_text is None:  # if tweet contains words less than two, ignore it
            continue

        if phase == 'train':
            images = tweet.split('\t')[3].split(',')
        else:
            images = tweet.split('\t')[4].split(',')
        for image in images:
            image = image.strip()
            if image in images_dict:
                label = tweet.split('\t')[-1].strip()
                if label == 'fake':
                    processed_tweet.append([original_tweet_text, tweet_text, images_dict[image], 0])
                else:
                    processed_tweet.append([original_tweet_text, tweet_text, images_dict[image], 1])

    pkl.dump(processed_tweet, open('C:/Users/singhbhu/Desktop/temp/' + phase + '_dataset.pkl', 'wb'))
    print(f"Tweets not in English in {phase} set:", ct)
    return


#if __name__ == '__main__':
process_dataset('train')
#process_dataset('test')