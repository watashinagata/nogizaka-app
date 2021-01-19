from fastapi import FastAPI, File, UploadFile
#from pydantic import BaseModel  # リクエストbodyを定義するために必要
#from typing import Optional,List  # ネストされたBodyを定義するために必要
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import joblib
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf
import base64

"""
def download_model():
    from keras.applications.vgg16 import VGG16
    base_model = VGG16(weights='imagenet', include_top=False, pooling="avg")
    #from keras.applications.xception import Xception
    #base_model = Xception(weights='imagenet', include_top=False, pooling="avg")
    return base_model
"""

app = FastAPI()
cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

@app.get("/")
def helloworld():
    return {"nogizaka":"app"}

@app.post("/")
async def create_file(file: bytes = File(...)):
    original_image = Image.open(BytesIO(file))
    array_img = np.array(original_image)

    face_list = cascade.detectMultiScale(array_img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))


    if len(face_list) != 0:
            try:
            for rect in face_list: # ひとつだけ顔を切り出して保存
                x = rect[0]
                y = rect[1]
                width = rect[2]
                height = rect[3]
                cropped_img = array_img[y:y + height, x:x + width]
                color=(0,0,255) #かこむ色(赤)
                new_original_image = cv2.cvtColor(array_img, cv2.COLOR_RGB2BGR)
                cv2.rectangle(new_original_image,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]),color,thickness = 2) #顔を囲む線
                break


            resize_img = cv2.resize(cropped_img, (100, 100))

            t = [image.img_to_array(array_img)]
            t = np.asarray(t)
            t = preprocess_input(t)
            #base_model = download_model() #学習済みモデルをkerasでダウンロードするパターン
            base_model = tf.keras.models.load_model('./vgg16.h5') #モデルを用意しておくパターン
            t_vgg16 = base_model.predict(t)
            clf = joblib.load('./clf.pkl')
            predict = clf.predict(t_vgg16)

            #return用に画像をbase64に変換
            _, dst_data = cv2.imencode('.jpg', new_original_image)
            new_original_image = base64.b64encode(dst_data)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            _, dst_data = cv2.imencode('.jpg', cropped_img)
            cropped_img = base64.b64encode(dst_data)

            if predict == [0]:
                code = 0
                text = 'nogizaka'
            elif predict == [1]:
                code = 1
                text = 'not nogizaka'
            except:
                code = 400
                text = 'error'
                new_original_image = ''
                cropped_img = ''
    else:
        code = 400
        text = 'error'
        new_original_image = ''
        cropped_img = ''

    return {
        'code': code,
        'text': text,
        'original_img': new_original_image,
        'cropped_img': cropped_img
    }

