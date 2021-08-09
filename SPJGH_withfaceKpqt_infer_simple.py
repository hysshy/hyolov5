import cv2
import time
import os
import torch
from faceKp_Qt_detect_one_roi import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = '/home/chase/shy/hyolov5/runs/train/exp15/weights/40last.pt'
model = load_model(weights, device)

testPath = '/home/chase/shy/yolodata/detect/images/val'
savePath = '/home/chase/shy/hyolov5/runs/train/exp15/draw'

face_classes = ['face','facewithmask']
person_classes = ['person', 'bicycle', 'ride', 'motorcycle']
pet_classes = ['dog', 'cat']
car_classes = ['car', 'truck']

colorList = [(28, 28, 28), (85, 26, 139), (255, 52, 179), (0, 255, 0), (25, 25, 112)]

for imgName in os.listdir(testPath):
    # test a single image and show the results
    print(imgName)
    # imgName = '2020-06-20 12:33:30_1202851740358782976.jpg'
    img = testPath +"/"+ imgName
    # img = '/home/chase/shy/yolodata/detect/images/train/180.jpg'
    img = cv2.imread(img)
    imgQt = img.copy()
    start = time.time()
    person_bboxes, person_labels, reidFeatsList, face_bboxes, face_labels, faceQts, faceKps, pets_bboxes, pets_labels, car_bboxes, car_labels = inference_detector(
        model, img, device)
    print(time.time() - start)
    for i in range(len(face_labels)):
        bbox = face_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = face_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, face_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)
            if label == 0:
                qt = round(float(faceQts[i]), 1)
                if not os.path.exists(savePath + '/' + str(qt)):
                    os.makedirs(savePath + '/' + str(qt))
                cv2.imwrite(savePath + '/' + str(qt) + '/' + imgName.replace('.jpg', str(i) + '.jpg'),
                            imgQt[bbox[1]:bbox[3], bbox[0]:bbox[2]])

            faceKp = faceKps[i].astype(int)
            clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            if len(bbox) > 0:
                for k in range(5):
                    point = (faceKp[k][0], faceKp[k][1])
                    cv2.circle(img, point, 3, clors[k], 0)

    for i in range(len(person_bboxes)):
        bbox = person_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = person_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, person_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

    for i in range(len(car_bboxes)):
        bbox = car_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = car_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, car_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)

    for i in range(len(pets_bboxes)):
        bbox = pets_bboxes[i].astype(int)
        if len(bbox) > 0:
            label = pets_labels[i]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i % len(colorList)], 1)
            cv2.putText(img, pet_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        colorList[i % len(colorList)], 2)
    #
    # for i in range(len(hat_bboxes_list)):
    #     bbox = hat_bboxes_list[i].astype(int)
    #     if len(bbox) > 0:
    #         label = hat_labels_list[i]
    #         cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colorList[i%len(colorList)], 1)
    #         cv2.putText(img, face_classes[label], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, colorList[i%len(colorList)], 2)

    cv2.imwrite(savePath + "/" + imgName, img)