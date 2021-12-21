import numpy as np
import PIL.Image
import requests
import json
URL = None


def init_params(config_file: str) -> dict:
    """
    Инициализирует параметры из json файла, преобразует необходимые поля к их типам, напр. str -> bool
    """
    with open(config_file, 'r') as json_params:
        params = json.load(json_params)
    return params


def load_image_file(file: str, mode: str = 'RGB') -> np.ndarray:
    """
    Загружает фото из файла и возвращает как numpy.ndarray
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


def get_recognition_report(image_array: np.ndarray = None, image_path: str = None, URL=None) -> dict:
    """
    API для работы с сервером, Посылает GET запрос на распознавание и возвращает dict (json)
    """
    if image_path:
        image_arr = load_image_file(image_path)
    else:
        image_arr = np.array(image_array, copy=True)
    image_json = json.dumps(image_arr.tolist())
    data = {'img_json': image_json}
    response = requests.get(URL, json=data)
    report = json.loads(response.content)
    return report


if __name__ == '__main__':
    config_file = "client_config.json"
    params = init_params(config_file)
    URL = params["URL"]

    image_path = "sample.png"
    image_arr = load_image_file(image_path)
    report = get_recognition_report(image_arr, URL=URL)
    print(report)
