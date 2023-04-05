pip install python-telegram-bot

import urllib.request
import numpy as np
import cv2
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

# загрузка модели YOLOv3
def load_model():
    net = cv2.dnn.readNet("./yolov3.weights", "./yolov3.cfg")
    classes = []
    with open("./coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, classes, colors, output_layers

# функция обработки изображения
def detect_objects(update: Update, context: CallbackContext) -> None:
    # получаем файл из сообщения
    file = context.bot.getFile(update.message.photo[-1].file_id)
    file_path = file.file_path
    # скачиваем файл
    urllib.request.urlretrieve(file_path, "image.jpg")
    # загружаем изображение
    img = cv2.imread("image.jpg")
    # получаем размеры изображения
    height, width, channels = img.shape
    # получаем пропорции для изменения размера картинки
    scale = 0.00392
    # изменяем размер картинки
    blob = cv2.dnn.blobFromImage(img, scale, (416,416), (0,0,0), True, crop=False)
    # загружаем модель YOLOv3
    net, classes, colors, output_layers = load_model()
    net.setInput(blob)
    # получаем результаты детектирования объектов
    outs = net.forward(output_layers)
    # создаем список найденных объектов
    class_ids = []
    confidences = []
    boxes = []
    # проходим по всем обнаруженным объектам
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # фильтруем объекты с низкой вероятностью
            if confidence > 0.5:
                # получаем координаты объектов
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                # добавляем объекты в список
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # применяем немаксимальное подавление
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # выводим результаты в сообщение
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            cv2.rectangle(img, (round(x), round(y)), (round(x+w), round(y+h)), color, 2)
            cv2.putText(img, label, (round(x)-10, round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # отправляем изображение с нарисованными прямоугольниками в чат
    cv2.imwrite("result.jpg", img)
    context.bot.send_photo(chat_id=update.message.chat_id, photo=open("result.jpg", "rb"))

# создаем объект бота и добавляем обработчик сообщений типа "фото"
updater = Updater(token='5639180492:AAEtOnYM3yCaHUl5xaA6TAI53LJUJDeMRd0', use_context=True)
updater.dispatcher.add_handler(MessageHandler(Filters.photo, detect_objects))

# запускаем бота
updater.start_polling()
updater.idle()
