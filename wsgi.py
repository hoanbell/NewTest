pip install -r requirements.txt
import tensorflow as tf
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from flask import Flask
from flask import jsonify, request
import numpy as np
import heapq
import cv2
graph = tf.compat.v1.get_default_graph()

# Load model from storage folder
# Download the Model from GitHub and path it in load_model()
model = load_model(r'NewTest/Saved Model')

# OUTPUT PREDICT
# Predict from an image
# 51 Categories
categories = ['alstroemeria', 'anemone', 'anthurium', 'arumlily', 'baloon flower', 'bellisdaisy', 'birdofparadise',
              'bouvardia', 'cherryblossom', 'coneflower',
              'cornflower', 'cypress', 'daffodil', 'dahlia', 'daisy', 'dandelion', 'dandelion',
              'edelweiss flower', 'foxglove', 'gazania',
              'hibicus', 'honeysuckle', 'hydrangea', 'iris', 'jasminum polyanthum', 'jasminum sambac', 'lantana',
              'laurel', 'lilac', 'lilies',
              'lilyofthevalley', 'lotus', 'loveinthemist', 'lupin', 'morningglory', 'myosotis', 'myrtus', 'orchid',
              'pansy', 'plumeria',
              'poinsettia', 'protea', 'ranunculus', 'rose', 'spearthistle', 'sunflower', 'tansy', 'tulip',
              'waterlilies', 'whiteclover', 'yarrow']


def topacc(num, imagepath):
    # Num is the Number of the highest predicted, NOT larger than 5
    # img = image.load_img(imagepath, target_size=(224, 224))
    # x = image.img_to_array(imagepath)
    x = np.expand_dims(imagepath, axis=0)
    imagepath = cv2.cvtColor(imagepath, cv2.COLOR_BGR2RGB)
    imagepath = cv2.resize(imagepath, (224, 224)).astype('float16')
    preds = model.predict(np.expand_dims(imagepath, axis=0))
    preds = np.array(preds).mean(axis=0)
    tops = sorted(preds, reverse=True)

    index5largest = heapq.nlargest(5, range(len(categories)), key=preds.__getitem__)
    labels = []
    hhshhhh = ""
    for i in range(num):
        hhshhhh += categories[index5largest[i]] + " : " + str(tops[i]) + "\n"

    return hhshhhh


# Input is a PATH of an image or a IMAGE_ULR
# def inputcall(inputin):
#     # print(inputin)
#     cap = cv2.VideoCapture(inputin)
#     if cap.isOpened():
#         _, img = cap.read()
#         cv2.imwrite('test.jpg', img)
#     img_path = 'test.jpg'
#     return img_path

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def uploadfile():
    file = request.files['image'].read()
    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    result = str(topacc(3, img))
    return result
if __name__ == '__main__':
    app.run(host='192.168.1.126')
