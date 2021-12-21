import easyocr


class TextRecognizer:
    def __init__(self, langs, gpu=False):
        try:
            self.reader = easyocr.Reader(langs, gpu=gpu)
        except RuntimeError as re:
            if gpu is True:
                print("Couldn't use GPU")
            else:
                print(re)
        except ValueError as ve:
            print(ve)

    def recognize(self, image):
        result = self.reader.readtext(image)
        report = [{'boxes': [[int(x) for x in pair] for pair in item[0]], 'text': item[1], 'confident': item[2]}
                  for item in result]
        return report
