# -*- coding: UTF-8 -*-

import matplotlib
matplotlib.use('Agg')
import os
import cv2
import torch
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.faceQt_general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
import numpy as np

def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model




def show_results(img, image_path, i, xywh, conf, landmarks, faceqt, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    # clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    # if int(class_num) in [8,9]:
    #     for i in range(5):
    #         point_x = int(landmarks[2 * i] * w)
    #         point_y = int(landmarks[2 * i + 1] * h)
    #         cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)
    #     faceqtRectPath = '/home/chase/shy/testyolo/yolov5/runs/train/exp68/faceqtrect'
    #     print(faceqt)
    #     faceqt = round(float(faceqt),1)
    #     if not os.path.exists(faceqtRectPath+'/'+str(faceqt)):
    #         os.makedirs(faceqtRectPath+'/'+str(faceqt))
    #     cv2.imwrite(faceqtRectPath+'/'+str(faceqt)+'/'+image_path.split('/')[-1].replace('.jpg','_'+str(i)+'.jpg'), img[y1:y2,x1:x2])

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def show_results2(img, image_path, i, bbox, conf, landmarks, faceqt, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1, y1 ,x2, y2 = bbox
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    if int(class_num) in [8,9]:
        for i in range(5):
            point_x = landmarks[2 * i]
            point_y = landmarks[2 * i + 1]
            cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)
        # faceqtRectPath = '/home/chase/shy/testyolo/yolov5/runs/train/exp68/faceqtrect'
        # print(faceqt)
        # faceqt = round(float(faceqt),1)
        # if not os.path.exists(faceqtRectPath+'/'+str(faceqt)):
        #     os.makedirs(faceqtRectPath+'/'+str(faceqt))
        # cv2.imwrite(faceqtRectPath+'/'+str(faceqt)+'/'+image_path.split('/')[-1].replace('.jpg','_'+str(i)+'.jpg'), img[y1:y2,x1:x2])

    tf = max(tl - 1, 1)  # font thickness
    label = str(int(class_num)) + ': ' + str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def pre_img(model, orgimg, device, img_size = 640):
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(orgimg, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img


def faceQt_detect(pred, faceId=[8,9]):
    faceIndex = torch.where((pred[0][:, -1] >= faceId[0]) & (pred[0][:, -1] <= faceId[1]))
    face_pred = pred[0][faceIndex]
    faceqt = face_pred[:,-2]
    return faceqt

def toResult(bboxes, labels, faceQts=None):
    personIndex = np.where(labels < 4)
    person_bboxes = bboxes[personIndex]
    person_labels = labels[personIndex]

    petIndex = np.where((labels >= 4) & (labels <= 5))
    pets_bboxes = bboxes[petIndex]
    pets_labels = labels[petIndex] -4

    carIndex = np.where((labels >= 6) & (labels <= 7))
    car_bboxes = bboxes[carIndex]
    car_labels = labels[carIndex] -6

    faceIndex = np.where((labels >= 8) & (labels <= 9))
    face_bboxes = bboxes[faceIndex]
    face_labels = labels[faceIndex] -8

    return person_bboxes, person_labels, None, face_bboxes, face_labels, faceQts, pets_bboxes, pets_labels, car_bboxes, car_labels

def inference_detector(model, orgimg, device, withfaceKp=True, withfaceQt = True, conf_thres = 0.3, iou_thres = 0.5):
    # Inference
    img = pre_img(model, orgimg, device)
    pred, _ = model(img)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)
    faceQts = faceQt_detect(pred, faceId=[8, 9]).cpu().numpy()
    det = pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
    return toResult(det[:, :5].cpu().numpy(), det[:, -1].int().cpu().numpy(), faceQts=faceQts)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = '/home/chase/shy/hyolov5/runs/train/exp/weights/85last.pt'
    model = load_model(weights, device)
    # image_path = '/home/chase/shy/dataset/spjgh/yolodata/images/train/1389.jpg'
    # detect_one(model, image_path, device)
    # print('over')
    image_path = '/home/chase/shy/yolodata/detect/images/train'
    for imgName in os.listdir(image_path):
        img = '/home/chase/shy/yolodata/detect/images/train/180.jpg'
        inference_detector(model, img, device)
        # detect_one(model, image_path+'/'+imgName, device)
