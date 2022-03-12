import os
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Concatenate, Dense, Input
from keras.models import Model
from keras.optimizers import Adam


def train(text_embedding_size, image_embedding_size, path):
    text = np.load('./temp/train_text.npy')
    im = np.load('./temp/train_image.npy')
    label = np.load('./temp/train_label.npy')

    test_text = np.load('./temp/test_text.npy')
    test_im = np.load('./temp/test_image.npy')
    test_label = np.load('./temp/test_label.npy')

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/tb'):
        os.makedirs(path + '/tb')
    if not os.path.exists(path + '/weights'):
        os.makedirs(path + '/weights')

    tensorboard = TensorBoard(log_dir=path + '/tb', write_graph=True, write_images=True)
    checkpoint = ModelCheckpoint(path + '/weights/{epoch:02d}.hdf5', monitor='loss', verbose=1, save_best_only=True,
                                 mode='auto')

    input_txt = Input(shape=(text_embedding_size,), name='input_txt')
    input_img = Input(shape=(image_embedding_size,), name='input_img')

    fc_txt_1 = Dense(768, activation='relu', name='fc_txt_1')(input_txt)
    fc_txt = Dense(32, activation='relu', name='fc_txt_2')(fc_txt_1)

    fc_img_1 = Dense(1536, name='fc_img_1', activation='relu')(input_img)
    fc_img = Dense(32, name='fc_img', activation='relu')(fc_img_1)

    h = Concatenate(axis=-1, name='concat')([fc_txt, fc_img])
    shared = Dense(35, name='shared', activation='relu')(h)
    res = Dense(1, name='output', activation='sigmoid')(shared)

    arch = Model(inputs=[input_txt, input_img], outputs=res)
    print(arch.summary())
    arch.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    arch.fit(x=[text, im], y=label, batch_size=128, epochs=300, callbacks=[checkpoint, tensorboard], shuffle=True,
             validation_data=([test_text, test_im], test_label))


if __name__ == '__main__':
    train(768, 1536, 'models')
