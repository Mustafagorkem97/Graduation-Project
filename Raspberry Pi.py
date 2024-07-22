import cv2
from picamera2 import Picamera2
import numpy as np


def configDNN():
    classNames = []
    classFile = "/home/raspberrypi/Desktop/Object_Detection_Files/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    # Nesnin tanımlanması için gereken modeli çekecek kod
    configPath = "/home/raspberrypi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "" / home / raspberrypi / Desktop / Object_Detection_Files / frozen_inference_graph.pb
    ""

    dnn = cv2.dnn_DetectionModel(weightsPath, configPath)
    dnn.setInputSize(320, 320)
    dnn.setInputScale(1.0 / 127.5)
    dnn.setInputMean((127.5, 127.5, 127.5))
    dnn.setInputSwapRB(True)

    return (dnn, classNames)


# thres = Bir nesne tespit edilmeden önceki güven eşiği
# nms = Maksimum Olmayan Bastırma- daha yüksek yüzde, algılanan örtüşen kutuların sayısını azaltır
# cup = tüm mevcut nesneler için algılanacak veya boşaltılacak nesnelerin adlarının listesi
def objectRecognition(dnn, classNames, image, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = dnn.detect(image, confThreshold=thres, nmsThreshold=nms)

    if len(objects) == 0:
        objects = classNames
    recognisedObjects = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                recognisedObjects.append([box, className])
                if (draw):
                    cv2.rectangle(image, box, color=(0, 0, 255), thickness=1)
                    cv2.putText(image, classNames[classId - 1] + " (" + str(round(confidence * 100, 2)) + ")",
                                (box[0] - 10, box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image, recognisedObjects


# Aşağıda Raspberry Pi OS'de görüntülenecek canlı yayın penceresinin boyutu belirlenir
if __name__ == "__main__":

    (dnn, classNames) = configDNN()

    picam2 = Picamera2()
    # Kamera formatını varsayılan RGBA yerine RGB olarak ayarlar
    config = picam2.create_preview_configuration({'format': 'RGB888'})
    picam2.configure(config)
    picam2.start()
    while (True):
        # Kamera görüntüsünü bir diziye kopyalar
        pc2array = picam2.capture_array()

        # Kamera görüntüsünü 180 derece döndürür
        pc2array = np.rot90(pc2array, 2).copy()

        # Nesne tanıma işlemini yapar
        result, objectInfo = objectRecognition(dnn, classNames, pc2array, 0.6, 0.6)

        # Bir pencerede gösterir
        cv2.imshow("Output", pc2array)
        cv2.waitKey(50)
