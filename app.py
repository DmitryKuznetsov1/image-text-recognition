from flask import Flask
from flask_restful import Api, Resource, reqparse
import json
import numpy as np
from text_recognizer import TextRecognizer
app = Flask(__name__)
api = Api(app)
text_recognizer = None


def init_params(config_file: str) -> dict:
    """
    Инициализирует параметры из json файла, преобразует необходимые поля к их типам, напр. str -> bool
    """
    with open(config_file, 'r') as json_params:
        params = json.load(json_params)
        if 'gpu' in params:
            params["gpu"] = True if params["gpu"].lower() == 'true' else False
    return params


class Quote(Resource):
    """
    Предоставляет обработку GET запросов, вычисляет время запроса, кодирует входное изображение в json и возвращает
     результат идентификации
    """
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('img_json')
        args = parser.parse_args()
        img_json = args['img_json']
        img_arr = np.array(json.loads(img_json), dtype=np.uint8)
        try:
            result = text_recognizer.recognize(img_arr)
            return result, 200
        except RuntimeError as re:
            return f'{re}', 404


if __name__ == '__main__':
    config_file = "app_config.json"
    params = init_params(config_file)

    text_recognizer = TextRecognizer(params["langs"])
    api.add_resource(Quote, "/api/recognize")
    app.run(debug=False)
