import requests
import json

def get_access_token():
    url = "https://api.interesturl.com/auth/login"
    datas = {
        "email": "lutergs@admin.com",
        "password": "admin-account"
    }

    response = requests.post(url, data=datas)
    return response


def upload_image(image_url, tag, access_token):
    url = "https://api.interesturl.com/images"
    header = {
        'Authorization' : 'Bearer ' + access_token
    }
    body = [
        {'url': image_url},
        {'tags': [{'name': tag}]}
    ]
    print(json.dumps(body))

    response = requests.post(url, headers=header, data=body)
    print("get response complete")
    return response


def get_image(id):
    url = "https://api.interesturl.com/images/1"
    id = {"id":"1"}
    response = requests.get(url, data=id)
    return response


if __name__ == "__main__":

    access_token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imx1dGVyZ3NAYWRtaW4uY29tIiwiaWF0IjoxNTg0NjEzMzU3LCJpc3MiOiJhbHBveGRldiJ9.9M7qaOjoCs3HtSyfYm0vOhG80bS2OSpaqrSoVChVARI'
    response = upload_image('http://image.msscdn.net/images/goods_img/20170817/604064/604064_1_220.jpg', "스트릿", access_token)
    #response = get_image(1)
    print(response)
    print(response.text)