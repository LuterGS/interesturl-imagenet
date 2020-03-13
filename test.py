import tensorflow as tf


test = tf.keras.datasets.fashion_mnist

(train_X, train_Y), (test_X, test_Y) = test.load_data()

print(train_X.shape, train_Y.shape)
print(train_X[0])





