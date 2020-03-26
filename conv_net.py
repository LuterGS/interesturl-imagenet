import tensorflow as tf
from keras import models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import vgg19
import os.path
import matplotlib.pyplot as plt

class conv_net:

    def __init__(self, pic_size, learning_rate, answer_len):
        # VGG-7정도의 얕은 신경망. 깊게 만들수는 있으나 그러려면 좀... (지금도 그램에서 신경망 init이 램 문제로 안되는거같은뎅...)
        self.nn_raw = tf.keras.Sequential([
            tf.keras.layers.Conv2D(input_shape=(pic_size, pic_size, 3), kernel_size=(3, 3), filters=32, padding='same',
                                   activation='relu'),  # input_shape 에서 이미지 사이즈 조정해야할듯
            # 첫 번째로 들어오는 신경망 필터, input_shape에서 마지막 3은 rgb 데이터로 들어올것이기 때문에 3으로, 커널 사이즈를 3,3으로 고정했지만 좀 더 크게 만들어야할수도 있음. 필터의 개수는 32개로
            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
            # 두 번째로 들어오는 신경망 필터. 64개의 필터 사용
            tf.keras.layers.Dropout(rate=0.2),
            # 과적합 방지를 위해 한 번의 dropout 설계

            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.2),
            # 위 과정을 한번 더 반복해주지만, 중간에 max_pooling 레이어 하나를 섞어준다.
            # 필터의 개수를 2배씩 늘려준다. VGG Style, 다르게 할 수도 있음.

            tf.keras.layers.Flatten(),
            # 다차원 신경망이기 떄문에 이걸 기본적인 신경망으로 바꾸기 위해 1차원 flatten layer 사용

            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(rate=0.2),
            # 2중 layer 설계. 중간에 과적합 방지를 위한 dropout layer 설계

            tf.keras.layers.Dense(units=20, activation='softmax')
            # 출력값 20개. 태그에 따른 20개로 결정
        ])
        self.nn_vgg19_lutergs = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(pic_size, pic_size, 3)),
            #tf.keras.layers.Conv2D(name='InputLayer', input_shape=[(pic_size, pic_size, 3)], kernel_size=(3, 3), filters=64, padding='same', activation='relu'),  # input_shape 에서 이미지 사이즈 조정해야할듯
                # VGG-19 Layer (InputLayer)

            # Block 1
            tf.keras.layers.Conv2D(name='block1_conv1', kernel_size=(3, 3), filters=64, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 1-1 (block1_conv1 (Conv2D))
            tf.keras.layers.Conv2D(name='block1_conv2', kernel_size=(3, 3), filters=64, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 1-2 (block1_conv2 (Conv2D))
            tf.keras.layers.MaxPooling2D(name='block1_pool', strides=(2, 2)),
                # VGG-19 Layer 1-3 (block1_pool (MaxPooling2D))

            # Block 2
            tf.keras.layers.Conv2D(name='block2_conv1', kernel_size=(3, 3), filters=128, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 2-1 (block2_conv1 (Conv2D))
            tf.keras.layers.Conv2D(name='block2_conv2', kernel_size=(3, 3), filters=128, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 2-2 (block2_conv2 (Conv2D))
            tf.keras.layers.MaxPooling2D(name='block2_pool', strides=(2, 2)),
                # VGG-19 Layer 2-3 (block2_pool (MaxPooling2D))

            # Block 3
            tf.keras.layers.Conv2D(name='block3_conv1', kernel_size=(3, 3), filters=256, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 3-1 (block3_conv1 (Conv2D))
            tf.keras.layers.Conv2D(name='block3_conv2', kernel_size=(3, 3), filters=256, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 3-2 (block3_conv2 (Conv2D))
            tf.keras.layers.Conv2D(name='block3_conv3', kernel_size=(3, 3), filters=256, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 3-3 (block3_conv3 (Conv2D))
            tf.keras.layers.Conv2D(name='block3_conv4', kernel_size=(3, 3), filters=256, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 3-4 (block3_conv4 (Conv2D))
            tf.keras.layers.MaxPooling2D(name='block3_pool', strides=(2, 2)),
                # VGG-19 Layer 3-5 (block3_pool (MaxPooling2D))

            # Block 4
            tf.keras.layers.Conv2D(name='block4_conv1', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 4-1 (block4_conv1 (Conv2D))
            tf.keras.layers.Conv2D(name='block4_conv2', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 4-2 (block4_conv2 (Conv2D))
            tf.keras.layers.Conv2D(name='block4_conv3', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 4-3 (block4_conv3 (Conv2D))
            tf.keras.layers.Conv2D(name='block4_conv4', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 4-4 (block4_conv4 (Conv2D))
            tf.keras.layers.MaxPooling2D(name='block4_pool', strides=(2, 2)),
                # VGG-19 Layer 4-5 (block4_pool (MaxPooling2D))

            # Block 5
            tf.keras.layers.Conv2D(name='block5_conv1', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 5-1 (block5_conv1 (Conv2D))
            tf.keras.layers.Conv2D(name='block5_conv2', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 5-2 (block5_conv2 (Conv2D))
            tf.keras.layers.Conv2D(name='block5_conv3', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 5-3 (block5_conv3 (Conv2D))
            tf.keras.layers.Conv2D(name='block5_conv4', kernel_size=(3, 3), filters=512, padding='same', strides=(1, 1), activation='relu'),
                # VGG-19 Layer 5-4 (block5_conv4 (Conv2D))
            tf.keras.layers.MaxPooling2D(name='block5_pool', strides=(2, 2)),
                # VGG-19 Layer 5-5 (block5_pool (MaxPooling2D))

            tf.keras.layers.Flatten(),
                # VGG-19 Layer Flatten

            tf.keras.layers.Dense(name='fc1', units=4096, activation='relu'),
            tf.keras.layers.Dropout(name='fc1_dropout', rate=0.2),
                # VGG-19 Layer fc1
                # VGG-19 Layer fc2
            tf.keras.layers.Dense(name='fc3', units=1024, activation='relu'),
            tf.keras.layers.Dropout(name='fc3_dropout', rate=0.2),
                # VGG-19 Layer fc3
            tf.keras.layers.Dense(name='fc4', units=512, activation='relu'),
                # VGG-19 Layer fc4
            tf.keras.layers.Dense(name='prediction', units=6, activation='softmax')
                # VGG-19 Layer fc5
        ])
        self.nn_vgg16_keras = VGG16(weights=None, input_shape=(pic_size, pic_size, 3), include_top=True, classes=answer_len)
        
        self.neural_network_layer = self.nn_vgg19_lutergs

        self.neural_network_layer.compile(
            optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        self.history = 0


    def train(self, input, answer, epochs, checkpoint_name, validation_split=0.25):
        checkpoint_path = 'training/' + checkpoint_name + ".ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1)

        self.history = self.neural_network_layer.fit(input, answer, epochs=epochs, validation_split=validation_split, callbacks=[checkpoint_callback])

    def test(self, input, answer):
        print(self.neural_network_layer.evaluate(input, answer, verbose=1))

