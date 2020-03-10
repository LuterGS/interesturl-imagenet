import csv
import image_process
import numpy as np

def tag_string_to_tag_int(tag_input):
    tag_jongryu = ['캐주얼', '스트릿', '이지 캐주얼', '유스', '댄디', '포멀', '스포츠', '레트로', '스쿨', '걸리시', '페미닌', '스포츠 캐주얼', '세미 포멀',
                   '데이트', '캠퍼스', '여행', '힙합', '시크', '로맨틱', '비즈니스']
    for i in range(20):
        if tag_input == tag_jongryu[i]:
            return i



f = open('/home/lutergs/Downloads/musinsa_tag_img.csv', 'r', encoding='euc-kr')
rdr = csv.reader(f)
dataset = []
for line in rdr:    # 총 40721개
    oneline = [tag_string_to_tag_int(line[0]), 'http://' + line[1]]
    dataset.append(oneline)
f.close()


print(dataset[100])
print(dataset[1000])
print(dataset[10000][1])
print(dataset[20000])
print(dataset[30000])

"""
for i in range(40720):
    print(dataset[i+1][1])
    dataset[i+1][1] = image_process.image_process.image_to_resized_numpy(dataset[i+1][1], 200)

t = open('/home/lutergs/Documents/musinsa.txt', 'a')
t.write('\n'.join(dataset))
t.close()
"""


