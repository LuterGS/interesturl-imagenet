from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import requests

class image_process:

    @staticmethod
    def image_resize(image_location, size):
        #이미지의 가로세로 크기 비율을 일정하게 줄이는 함수. image_to_resized_numpy의 기본 속성에 따라 최대길이 200으로 비율 맞게 리사이징됨.
        raw_import = load_img(image_location)
        raw_import.thumbnail((size, size))
        return raw_import

    @staticmethod
    def image_add_padding(image_data, size):
        #이미지가 정사각형 이미지가 아닐 때 패딩 추가해주는 함수.
        col = int((size - image_data.size[0]) / 2)
        row = int((size - image_data.size[1]) / 2)
        padding_image = Image.new("RGB", (size, size))
        padding_image.paste(image_data, (col, row))
        return padding_image

    @staticmethod
    def image_to_numpy(image_data):
        """
        :param image_data: 이미지의 PIL.image로 열은 값.
        :return: 넘파이 어레이
        """
        n_array = img_to_array(image_data)
        n_array = n_array.reshape((1,) + n_array.shape)
        n_array = n_array/255.0
        return n_array[0]


    @staticmethod
    def image_to_numpy_loc(image_loc):
        image_data = load_img(image_loc)
        n_array = img_to_array(image_data)
        n_array = n_array.reshape((1,) + n_array.shape)
        n_array = n_array / 255.0
        return n_array


    @staticmethod
    def image_to_resized_numpy(image_location, size=200, save_location=0):
        """
        :param image_location: 이미지가 저장되어있는 위치 (웹에서 받아오는거 처리도 되게끔 해야겠는걸
        :param size: 이미지가 정사각형 형태로 리사이징되는데, 그때의 한 변의 픽셀 개수
        :param save_location: 혹시나 크롭한 이미지를 저장할 위치를 지정하면 저장해줌.
        :return: 이미지를 넘파이 배열로 변환한 값. 4차원 배열이 주어지지만 배열의 첫 번째 차원은 1이므로, 사실상 가로/세로/rgb3의 3차원으로 나온다 볼 수 있음.
        """
        resized_image = image_process.image_resize(image_location, size)
        padded_image = image_process.image_add_padding(resized_image, size)
        resized_numpy = image_process.image_to_numpy(padded_image)
        if save_location != 0:
            padded_image.save(save_location)
        return resized_numpy