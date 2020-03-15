import conv_net
import data_setup
import matplotlib.pyplot as plt
import os.path

pic_size = 100
answer_len = 6
#사진 리사이징 한 변의 길이. 많은 변수를 좌지우지하므로 메인에서 관

train_input, train_answer, test_input, test_answer = data_setup.fetch_data(pic_size, 'gagu')
print("Get ALL data complete\n")
print(train_input.shape)

train_model = conv_net.conv_net(pic_size, 0.0001, answer_len)
print("Set NN Model complete\n")

train_model.neural_network_layer.summary()

train_model.train(train_input, train_answer, 10, 'checkpoint')
print("Train Model complete\n")


#draw graph
plt.figure(figsize=(12,4))

plt.subplot(1, 2, 1)
plt.plot(train_model.history.history['loss'], 'b-', label='loss')
plt.plot(train_model.history.history['val_loss'], 'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_model.history.history['accuracy'], 'g-', label='accuracy')
plt.plot(train_model.history.history['val_accuracy'], 'k--', label='val_accuracy')
plt.xlabel('Epoch')
plt.ylim(0.7, 1)
plt.legend()

plt.show()


train_model.test(test_input, test_answer)
print("Test Model complete\n")


