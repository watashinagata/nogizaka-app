FROM python:3.6

# update pip and conda
#RUN pip install --upgrade pip

RUN pip install --upgrade pip && pip install \
Tensorflow-cpu \
keras \
fastapi \
uvicorn \
joblib \
pillow \
numpy \
python-multipart



# opencv
RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install  opencv-python

RUN pip install sklearn

COPY . .

# 起動
CMD ["uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
