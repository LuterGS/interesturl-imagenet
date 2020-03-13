import conv_net
import data_setup
import matplotlib.pyplot as plt
import os.path

pic_size = 100
#사진 리사이징 한 변의 길이. 많은 변수를 좌지우지하므로 메인에서 관

array_path = 'tag_test.npz'
if os.path.isfile(array_path):
    pass
else :
    data_setup.save_data(pic_size, 0.8)
train_input, train_answer, test_input, test_answer = data_setup.load_data()
print("Get ALL data complete\n")

train_model = conv_net.conv_net(pic_size)
print("Set NN Model complete\n")

train_model.train(train_input, train_answer, 25)
print("Train Model complete\n")


#draw graph
plt.figure(figsuze=(12,4))

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
print("Test Model completeb\n")
