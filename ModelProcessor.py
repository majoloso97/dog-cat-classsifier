from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import requests
from tempfile import NamedTemporaryFile
from PIL import Image
from io import BytesIO

class ModelProcessor:
    def __init__(self, source_url, extension = 'h5'):
        self.model = self.load_model(source_url, extension)
        self.sample_images = {}


    def get_from_url(self, source_url):
        response = requests.request('GET', url = source_url, allow_redirects = True)
        if response.status_code == 200:
            return response.content


    def load_model(self, source_url, extension='h5'):
        with NamedTemporaryFile(suffix = extension) as tmp:
            content = self.get_from_url(source_url)
            tmp.write(content)
            tmp.read()
            model = keras.models.load_model(tmp.name)
            return model


    def load_image(self, source, from_url = True, image_size = (180,180)):
        if from_url:
            image = Image.open(
                            BytesIO(
                                self.get_from_url(source)
                                )
                            )
            image = image.resize(image_size)        
        else:
            image = keras.preprocessing.image.load_img(source, target_size=image_size)
        return image


    def preprocess_image(self, image):
        img_array = keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)
        return img_array


    def show_img(self, image):
        figure, axes = plt.subplots()
        axes.imshow(image)
        plt.axis("off")
        
        return figure

    def show_predict_image(self, source, from_url = True, image_size = (180,180)):
        image = self.load_image(source, from_url=from_url, image_size=image_size)
        plot = self.show_img(image)
        prep_image = self.preprocess_image(image)

        predictions = self.model.predict(prep_image)
        score = predictions[0]
        
        predicted = "The selected image is %.2f percent cat and %.2f percent dog."% (100 * (1 - score), 100 * score)
        return predicted, plot