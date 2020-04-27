#! /usr/bin/env python
# encoding: utf-8

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec

def TextCNN(train_data,train_label,val_data,val_label,sequence_length,batch_size,epochs,embedding_matrix):
    
    main_input = keras.Input(shape=(sequence_length,), dtype='float32')
    # 词嵌入层
    embedder = keras.layers.Embedding(input_dim=embedding_matrix.shape[0],
                                      output_dim=embedding_matrix.shape[1],
                                      input_length=sequence_length,
                                      weights=[embedding_matrix],
                                      trainable=False)
    embed = embedder(main_input)
    # 卷积层
    cnn1 = keras.layers.Conv1D(256, 3, 
                               padding='same', strides=1, 
                               activation='relu')(embed)
    cnn1 = keras.layers.MaxPooling1D(pool_size=48)(cnn1)
    cnn2 = keras.layers.Conv1D(256, 4, 
                               padding='same', strides=1, 
                               activation='relu')(embed)
    cnn2 = keras.layers.MaxPooling1D(pool_size=47)(cnn2)
    cnn3 = keras.layers.Conv1D(256, 5, 
                               padding='same', strides=1, 
                               activation='relu')(embed)
    cnn3 = keras.layers.MaxPooling1D(pool_size=46)(cnn3)
    cnn = keras.layers.concatenate([cnn1, cnn2, cnn3], axis=-1)
    # 输出层
    flat = keras.layers.Flatten()(cnn)
    drop = keras.layers.Dropout(0.2)(flat)
    main_output = keras.layers.Dense(14, activation='softmax')(drop)
    model = keras.Model(inputs=main_input, outputs=main_output)
    
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    keras.utils.plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True,rankdir='TB')
    
    history = model.fit(train_data, 
                        train_label, 
                        validation_data=(val_data, val_label), 
                        batch_size=batch_size, 
                        epochs=epochs)
    
    model.save('model.h5')
    
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label = 'val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    
    test_loss, test_acc = model.evaluate(val_data,
                                         val_label,
                                         verbose=2)

def load_data(path):
    data_array = path

    array = np.load(data_array)
    np.random.shuffle(array)
    
    train_array = array[0:24000]
    val_array = array[24000:]
    
    train_label = train_array[:,0:1]
    train_data = train_array[:,1:]
    
    val_label = val_array[:,0:1]
    val_data = val_array[:,1:]
    print('load data complete!')
    return train_data,train_label,val_data,val_label
    
def main():
    data_array = 'data/array/labels_words.npy'
    word2vec_path = 'model/word2vec.model'
    sequence_length = 300 # 最大句子长度
    epochs = 20  # 迭代次数
    batch_size = 128 # 批大小

    train_data,train_label,val_data,val_label = load_data(data_array)
    
    w2vModel = Word2Vec.load(word2vec_path)
    embedding_matrix = w2vModel.wv.vectors
    
    TextCNN(train_data,
            train_label,
            val_data,
            val_label,
            sequence_length,
            batch_size,
            epochs,
            embedding_matrix)

if __name__=="__main__":
    main()