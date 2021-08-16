import cv2
import sys
import time
import numpy as np

import json
import base64
import requests

import boto3

service_name = 's3'
endpoint_url = 'https://kr.object.ncloudstorage.com'
region_name = 'kr-standard'
accesss_key = 'OTOSP4zl6AHejEc10Bj7'
secret_key = 'FNzp5yHsjUgIGa2Uk9uTTZTRUZ4bGAPEWxhtjtiL'

s3= boto3.client(service_name, endpoint_url=endpoint_url, aws_access_key_id=accesss_key,
                     aws_secret_access_key=secret_key)

bucket_name = 'dwkrefrigerator'
    
#object_name = 'table_image'
#local_file_path = './input_image/input.png'

class CameraDetection:
    
    def __init__(self):
        self.frameWidth = 640   #카메라 크기 초기화
        self.frameHeight = 480
        #self.frameWidth, self.frameHeight = self.frame_size_check()    #카메라 사이즈 체크
        self.frame = np.zeros((self.frameHeight, self.frameWidth), np.uint8)    #카메라 프레임 초기화
        self.approx_points = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32) #꼭지점 배열 초기화
        
        self.srcQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspective source 점 4개 초기화
        self.dstQuad = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], np.float32)   #Perspecrive 점 4개 초기화
        
        self.dstWidth = 0   #펼칠 프레임 크기 초기화
        self.dstHeight = 0
        self.receipt_list = []
    #프레임 크기 확인
    def frame_size_check(self):
        capture = cv2.VideoCapture(0)   #카메라 가져오기
        if not capture.isOpened():      #카메라 안열리면 종료
            print('Camera open failed!')
            sys.exit()
        frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))     #프레임 크기 확인
        frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frameSize = (frameWidth, frameHeight)
        print("Frame Size : {}".format(frameSize))  
        capture.release()
        return frameWidth, frameHeight
    
    #외곽선 만들기
    def make_contours(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #(전처리) - 그레이스케일
        frame_unnoise = cv2.bilateralFilter(frame_gray, -1, 10, 5)  #(전처리) - 양방향 필터
        _, frame_binary = cv2.threshold(frame_unnoise, 0, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)   #이진화 - OTSU알고리즘
        contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #외곽선 검출 - 외곽 검출
        return contours
    
    #사각형 찾기
    def find_rectangle(self, frame, contours):
        for points in contours:
            if cv2.contourArea(points) < 1000:  #크기가 1000이하인 contour들은 무시
                continue
            approx = cv2.approxPolyDP(points, cv2.arcLength(points, True) * 0.02, True) #외곽선 근사화
            if len(approx) != 4:    #사각형이 아닌것들은 무시
                continue
            self.approx_points = approx
            cv2.polylines(frame, points, True, (0, 0, 255), thickness = 3)  #외곽선 빨간색 굵기 3 선으로 표시
            if cv2.contourArea(points) > self.frameWidth * self.frameHeight * 0.8: #일정 크기 이상이면 찾기 스탑
                return True         

            
    def reorder_points(self, points):
        idx = np.lexsort((points[:, 1], points[:, 0])) #x좌표로 먼저 정렬 후 y좌표로 정렬
        points = points[idx] #정렬
        
        if points[0, 1] > points[1, 1]:
            points[[0, 1]] = points[[1, 0]] #0번이 1번보다 아래에 있다면 교체
            
        if points[2, 1] < points[3, 1]:
            points[[2, 3]] = points[[3, 2]] #2번이 3번보다 아래에 있다면 교체
        
        self.dstWidth = round(2 * (points[3, 0] - points[0, 0]))
        self.dstHeight = round(2 * (points[1, 1] - points[0, 1]))   #결과 프레임 크기 설정
        
        self.dstQuad = np.array([[0, 0], [0, self.dstHeight], 
                                 [self.dstWidth, self.dstHeight], [self.dstWidth, 0]], np.float32) #결과 프레임 크기 설정
        
        return points
    
    def make_return_image(self, frame):
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    #전처리 그레이스케일
        image_blur = cv2.GaussianBlur(image_gray, (3, 3), 0)    #전처리 가우시안 필터
        #cv2.imshow('image_blur', image_blur)
        #cv2.waitKey()
        return image_blur

    def camera_detect(self):
        capture = cv2.VideoCapture(0)   #웹캠 켜기
        if not capture.isOpened():
            print('Camera open failed!')
            sys.exit()
        
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)  
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)    #웹캠마다 프레임 달라질수 있으니 프레임 고정
        
        while True:
            ret, self.frame = capture.read()    #웹캠에서 프레임 가져오기
            if not ret:
                print("Frame read error!")
                sys.exit()
    
            contours = self.make_contours(self.frame)   #외곽선 검출
            
            is_rectangle = self.find_rectangle(self.frame, contours)    #사각형 확인
            
            if is_rectangle == True:    #사각형이면 멈추기
                count = 1
                self.srcQuad = self.reorder_points(self.approx_points.reshape(4, 2).astype(np.float32)) #꼭지점 정렬
                perspective = cv2.getPerspectiveTransform(self.srcQuad, self.dstQuad)   #Perspective 행렬 반환
                dst_frame = cv2.warpPerspective(self.frame, perspective, (self.dstWidth, self.dstHeight))   #Perspective 변환
                cv2.imwrite('./cloud/receipt_image.png', dst_frame)
                s3.upload_file('./cloud/receipt_image.png', bucket_name, 'receipt_image')
                timestamp = round(time.time())
                ret, buffer = cv2.imencode('.png', dst_frame)   #이미지를 png형태로 변환
                png_as_text = base64.b64encode(buffer)  #이미지를 b64로 인코딩
    
                x_ocr_secret = "ZlBWaUFqdFp4bVFZTUpnTHZucGVwZFVOZUt3cllVY0o="
                ocr_invoke_url = "https://f04b1d88f83e41a6b1df8ce399b61fb5.apigw.ntruss.com/custom/v1/9782/eed9df9be97433637c2950146e96ff0956759904a38b1e5e6a9a69a7f6691a68/general"
                uuid = "ce3e4c43-4c69-4188-ba2a-8ad24d9615e4"
    
                headers = {
                    "X-OCR-SECRET" : x_ocr_secret,
                    "Content-Type" : "application/json"
                    }
                
                data = {
                        "version" : "V1",
                        "requestId" : uuid,
                        "timestamp" : timestamp,
                        "images" : [
                            {
                                "format" : "png",
                                "data" : png_as_text.decode('utf-8'),
                                "name" : "sample_image_{}".format(count)
                                }
                            ]
                        }
    
                data = json.dumps(data) #파이썬 객체를 json로 변환
                response = requests.post(ocr_invoke_url, headers=headers, data=data)    #API로 보내기
                res = json.loads(response.text) #json을 객체로 변환
                
                res_array = res.get('images')
                file = open('./cloud/food_list.txt', 'w')
                food_label = open('./cloud/local/food_label.txt', 'r', encoding='utf-8')
                food_label_lines = food_label.readlines()
                for f_list in res_array[0].get('fields'):
                    for line in food_label_lines:
                        line = line.rstrip('\n')
                        if (f_list.get('inferText')).find(line) > -1:
                            self.receipt_list.append(line)
                self.receipt_list = list(set(self.receipt_list))
                for rec_list in self.receipt_list:
                    print(rec_list)
                    file.write(rec_list)
                    file.write('\n')
                file.close()
                food_label.close()
                s3.upload_file('./cloud/food_list.txt', bucket_name, 'table_list')
                
            cv2.imshow('Frame', self.frame) #프레임 출력
            cv2.imwrite('./cloud/table_frame.png', self.frame)
            s3.upload_file('./cloud/table_frame.png', bucket_name, 'table_frame')
            if cv2.waitKey(33) == ord('q'): #33ms마다 프레임 바꾸기 q누르면 멈춤
                break
        
        capture.release()   #웹캠 객체 풀기
        cv2.destroyAllWindows() #모든 창 닫기
#        return dst_frame  #결과 프레임 출력
#        return self.make_return_image(dst_frame)


detection = CameraDetection()
detection.camera_detect()
cv2.waitKey()
cv2.destroyAllWindows()