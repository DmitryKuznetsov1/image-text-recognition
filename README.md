# image-text-recognition
**Web-сервис состоит из:**

* `app.py` — локальный сервер на Flask, который принимает запросы через экземпляр класса 
  `TextRecognizer`;

* `FaceIdentifier.py` — содержит класс распознавателя с единственным методом `recognize(image)`;

* `client.py` — клиент для работы с сервером.
  Внутри функция для взаимодействия с Api,
  можно передать значение как numpy массив или как путь до файла.
  
**Логика:**
1. Сначала запускается сервер `app.py` (в данном примере локально по адресу `http://127.0.0.1:5000`). Там же
создается экземпляр класса TextRecognition (распознаваемые языке перечислены в `app_config.json`);


2. После этого на стороне клиента можно открыть изображение по его пути или получить извне массивом. Далее массив сериализуется
с помощью `json` и отправляется как параметр вместе с `get` запросом по адресу `http://127.0.0.1:5000/api/recognize`;


3. На выходе получаем json со следующей информацией `{'bounding box': ..., 'text': ...,
   'confident': ...}` для каждой распознанной еденицы текста.`
   
# Пример
Для начала необходимо запустить сервер -- app.py, в данном случае -- локально (и не в ноутбуке), GET запрос идет по URL, который можно настроить resources/config.json, поумолчанию: http://127.0.0.1:5000/


```python
!python3 app.py
```

    Using CPU. Note: This module is much faster with a GPU.
     * Serving Flask app 'app' (lazy loading)
     * Environment: production
    [31m   WARNING: This is a development server. Do not use it in a production deployment.[0m
    [2m   Use a production WSGI server instead.[0m
     * Debug mode: off
     * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
    ^C



```python
import client
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

%matplotlib inline
%config InlineBackend.figure_format = 'svg' # 'svg', 'retina'
plt.style.use('seaborn-white')
```


```python
def repr_rect(points):
    xy = tuple(points[0])
    w = points[1][0] - points[0][0]
    h = points[3][1] - points[0][1]
    return xy, w, h

def repr_report(image_arr, report):
    fig, ax = plt.subplots()
    ax.imshow(image_arr)
    for el in report:
        points = el['boxes']
        xy, w, h = repr_rect(points)
        rect = patches.Rectangle(xy, w, h, linewidth=1.5, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        print(f"{el['text']}:", round(el['confident'], 2))
    plt.show()
```

Мнимальные настройки


```python
config_file = "client_config.json"
params = client.init_params(config_file)
URL = params["URL"]
URL
```




    'http://127.0.0.1:5000/api/recognize'



Теперь можно проверить систему в действии


```python
image_path = "sample.png"
image_arr = client.load_image_file(image_path)
report = client.get_recognition_report(image_arr, URL=URL)
```


```python
repr_report(image_arr, report)
```

    IMAGE: 0.99
    TO TEXT:: 0.51
    HOW: 0.26
    TO: 0.76
    EXTRACT: 0.97
    TEXT: 1.0
    FROM: 0.89
    AN IMAGE: 0.83




![svg](example_files/example_8_1.svg)




```python

```
