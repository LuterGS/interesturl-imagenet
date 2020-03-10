import tensorflow as tf
import matplotlib.pyplot as plt

class conv_net:

    def __init__(self):
        # VGG-7정도의 얕은 신경망. 깊게 만들수는 있으나 그러려면 좀... (지금도 그램에서 신경망 init이 램 문제로 안되는거같은뎅...)
        self.neural_network_layer = tf.keras.Sequential([
            tf.keras.layers.Conv2D(input_shape=(200, 200, 3), kernel_size=(3, 3), filters=32, padding='same',
                                   activation='relu'),  # input_shape 에서 이미지 사이즈 조정해야할듯
            # 첫 번째로 들어오는 신경망 필터, input_shape에서 마지막 3은 rgb 데이터로 들어올것이기 때문에 3으로, 커널 사이즈를 3,3으로 고정했지만 좀 더 크게 만들어야할수도 있음. 필터의 개수는 32개로
            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
            # 두 번째로 들어오는 신경망 필터. 64개의 필터 사용
            tf.keras.layers.Dropout(rate=0.5),
            # 과적합 방지를 위해 한 번의 dropout 설계

            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
            tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(rate=0.5),
            # 위 과정을 한번 더 반복해주지만, 중간에 max_pooling 레이어 하나를 섞어준다.
            # 필터의 개수를 2배씩 늘려준다. VGG Style, 다르게 할 수도 있음.

            tf.keras.layers.Flatten(),
            # 다차원 신경망이기 떄문에 이걸 기본적인 신경망으로 바꾸기 위해 1차원 flatten layer 사용

            tf.keras.layers.Dense(units=512, activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.Dropout(rate=0.5),
            # 2중 layer 설계. 중간에 과적합 방지를 위한 dropout layer 설계

            tf.keras.layers.Dense(units=20, activation='softmax')
            # 출력값 20개. 태그에 따른 20개로 결정
        ])
        self.neural_network_layer.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        self.history = 0

    def train(self, input, answer, epoch, validation_split=0.25):
        self.history = self.neural_network_layer.fit(input, answer, epoch, validation_split)

    def test(self, input, answer):
        self.neural_network_layer.evaluate(input, answer, verbose=0)




