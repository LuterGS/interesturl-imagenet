import image_process
import numpy as np
import random
import os.path
import matplotlib.pyplot as plt

"""
데이터셋 개수 (cloth)

걸리시 935개 2092
댄디 854개 1739
데이트 169개 311
레트로 116개 243
로맨틱 238개 721
비즈니스 165개 440
세미 포멀 443개 886
스쿨 188개 373
스트릿 4329개 9040
스포츠 캐주얼 383개 692
스포츠 631개 1193
시크 181개 523
여행 128개 246
유스 394개 693
이지 캐주얼 2569개 5118
캐주얼 8282개 11586
캠퍼스 434개 934
페미닌 2개 2
포멀 1895개 3559
힙합 30개 70
"""

cloth_tag_list = ['걸리시', '댄디', '데이트', '레트로', '로맨틱', '비즈니스', '세미 포멀', '스쿨', '스트릿', '스포츠 캐주얼', '스포츠', '시크', '여행', '유스', '이지 캐주얼', '캐주얼', '캠퍼스', '페미닌', '포멀', '힙합']
cloth_tag_maxnum = [2092, 1739, 311, 243, 721, 440, 885, 373, 9040, 692, 1193, 523, 246, 693, 5118, 11586, 934, 2, 3559, 70]
cloth_tag_maxnum_adjusted = [800, 700, 311, 243, 721, 440, 885, 373, 800, 692, 800, 523, 246, 693, 800, 800, 934, 2, 800, 70]


gagu_tag_list = ['내츄럴', '럭셔리', '모던&심플', '북유럽', '빈티지', '클래식']
gagu_tag_maxnum = [3188, 899, 6471, 446, 2897, 1864]

def fetch_data(pic_size, data='cloth'):
    array_path = data + '_tag_data.npz'
    if os.path.isfile(array_path):
        pass
    else:
        print("No original data found, will fetch data from raw image\n")
        save_data(pic_size, 0.8, data)
    train_input, train_answer, test_input, test_answer = load_data(data)
    return train_input, train_answer, test_input, test_answer


def save_data(pic_size, test_ratio, data='cloth'):
    train_input, train_answer, test_input, test_answer = get_test_train_data(pic_size, test_ratio, data)
    np.savez(data + '_tag_data.npz', train_input=train_input, train_answer=train_answer, test_input=test_input, test_answer=test_answer)


def load_data(data='cloth'):
    all = np.load(data + '_tag_data.npz')
    train_input = all['train_input']
    train_answer = all['train_answer']
    test_input = all['test_input']
    test_answer = all['test_answer']
    return train_input, train_answer, test_input, test_answer


def get_test_train_data(pic_size, test_ratio, data):
    input, answer = get_data(pic_size, 'list', data=data)
    print(len(answer))
    train_ratio = int(len(answer) * test_ratio)
    train_input, train_answer = input[0:train_ratio], answer[0:train_ratio]
    test_input, test_answer = input[train_ratio:-1], answer[train_ratio:-1]
    train_input = np.asarray(train_input, dtype=np.float32)
    train_answer = np.asarray(train_answer)
    test_input = np.asarray(test_input, dtype=np.float32)
    test_answer = np.asarray(test_answer)
    return train_input, train_answer, test_input, test_answer



def get_data(pic_size, type='list', data="cloth"):
    input, output, raw = [], [], []

    if data == "cloth":
        output_len = 20
        tag_maxnum = cloth_tag_maxnum
        tag_list = cloth_tag_list
        file_path = 'test_data'
    elif data == "gagu":
        output_len = 6
        tag_maxnum = gagu_tag_maxnum
        tag_list = gagu_tag_list
        file_path = 'gagu_data'

    print(tag_maxnum, tag_list)

    for i in range(output_len):
        for j in range(tag_maxnum[i]):
            to_string = str(j + 1)
            #print('/home/lutergs/Documents/' + file_path + '/' + tag_list[i] + '_' + to_string + '.jpg')
            try:
                raw.append([image_process.image_process.image_to_resized_numpy(
                    '/home/lutergs/Documents/' + file_path + '/' + tag_list[i] + '_' + to_string + '.jpg', pic_size, '/home/lutergs/Documents/gagu_test/' + tag_list[i] + to_string + '.jpg'), i])
            except FileNotFoundError:
                pass

    print(len(raw))
    random.shuffle(raw)
    for i in range(len(raw)):
        input.append(raw[i][0])
        output.append(raw[i][1])


    if type == 'array':
        input_np = np.asarray(input, dtype=np.float32)
        output_np = np.asarray(output)
        return input_np, output_np

    return input, output



def test_d():
    input, output, raw_data = [], [], []
    raw_data.append([image_process.image_process.image_to_resized_numpy('/home/lutergs/Documents/gagu_data/내츄럴_1.jpg', 200, '/home/lutergs/Documents/gagu_test/내츄럴_1.jpg'), 0])
    random.shuffle(raw_data)

    for i in range(len(raw_data)):
        input.append(raw_data[i][0])
        output.append(raw_data[i][1])
    input_np = np.asarray(input, dtype=np.float32)
    output_np = np.asarray(output)
    return input_np, output_np



if __name__ == "__main__":
    input, output = test_d()

    plt.figure(figsize=(10,10))
    plt.imshow(input[0], cmap='gray')
    plt.show()


    print(input.shape, input[0], output.shape, output)