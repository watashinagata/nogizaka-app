import streamlit as st
import requests
from PIL import Image
import iof
import base64
from config import *

st.title('あなたが乃木坂かどうか判定するAIだよ')
uploaded_file = st.file_uploader('顔の写った写真をください', type=["jpg"], accept_multiple_files=False)

if uploaded_file is not None:
    tmpimg = Image.open(uploaded_file)

    with io.BytesIO() as output:
        tmpimg.save(output,format="JPEG")
        contents = output.getvalue()#バイナリ取得

    files = {'file': contents}
    # files引数に指定します.
    r = requests.post(nogizaka_api, files=files)
    #r = requests.post('http://localhost:8080/', files=files) #ローカルでapi立ち上げた時用
    label = r.json()['code']
    if label == 0:
        st.title('おめでとう!!!!!!')
        st.title('あなたは乃木坂だよ！')

        st.subheader('顔と認識された部分')
        col1, col2 = st.beta_columns(2)
        img =  Image.open(io.BytesIO(base64.b64decode(r.json()['original_img'])))
        col1.image(img,use_column_width=True)
        img =  Image.open(io.BytesIO(base64.b64decode(r.json()['cropped_img'])))
        col2.image(img,use_column_width=True)
    elif label == 1:
        st.title('残念!!!!!!')
        st.title('あなたは乃木坂じゃない*よ...')

        st.subheader('顔と認識された部分')
        col1, col2 = st.beta_columns(2)
        img =  Image.open(io.BytesIO(base64.b64decode(r.json()['original_img'])))
        col1.image(img,use_column_width=True)
        img =  Image.open(io.BytesIO(base64.b64decode(r.json()['cropped_img'])))
        col2.image(img,use_column_width=True)
    else:
        st.title('画像から顔が見つからなかったよ...')
        st.title('違う画像でためしてみて...')