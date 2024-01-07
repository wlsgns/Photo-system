import collections
import numpy as np
from PIL import Image
from mtcnn import mtcnn
from sklearn.cluster import DBSCAN
import os
import face_recognition as fr
import cv2
import math
import time
import keyboard
import glob
import pickle
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
import shutil

model_pose = YOLO('yolov8n-pose.pt')

def face():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        print("WebCam is not running")
        exit()

    # 변수 초기화
    time_num = 0
    image_num = 0

    # 얼굴 이미지를 저장할 디렉토리 경로
    save_directory = r"face"

    # 웹캠 프레임 읽기
    while webcam.isOpened():
        status, frame = webcam.read()
        time_num = time_num + 1

        if not status:
            break

        # 웹캠 화면 출력
        cv2.imshow("WebCam", frame)

        # -- 캡쳐 프레임 간격 설정
        if time_num == 90:
            image_num = image_num + 1

            # 프레임 저장
            cv2.imwrite('org/img' + str(image_num) + '.jpg', frame)

            # 저장된 프레임 불러오기
            image_path = os.path.join(os.getcwd(), 'org/img' + str(image_num) + '.jpg')
            image = fr.load_image_file(image_path)

            # 얼굴 위치 정보 찾기
            face_locations = fr.face_locations(image)

            # 감지된 얼굴의 수 출력
            print("이 사진에서 {}개의 얼굴을 찾았습니다.".format(len(face_locations)))

            # 각 얼굴에 대한 처리
            for i, face_location in enumerate(face_locations):
                # 얼굴 위치 정보 변수
                top, right, bottom, left = face_location

                # 이미지 파일의 이름 생성
                new_file_name = f"face_{image_num}_{i}.jpg"

                # 이미지 파일의 전체 경로
                save_path = os.path.join(save_directory, new_file_name)

                # 얼굴 이미지 추출 및 저장
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                pil_image.save(save_path)

                # 결과 출력
                print(f"얼굴 {i}을 저장했습니다. 경로: {save_path}")

            # 다음 얼굴을 위해 변수 업데이트
            image_num += 1
            time_num = 0

        # -- q 입력시 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 웹캠 해제 및 창 닫기
    webcam.release()
    cv2.destroyAllWindows()

def move_image(image, id, labelID):
    path = 'cluster/label' + str(labelID)
    print(path)
    if os.path.exists(path) == False:
        os.mkdir(path)

    filename = str(id) + '.jpg'

    cv2.imwrite(os.path.join(path, filename), image)

    return

def embedding(self,i):
    d = []
    self = cv2.imread(self)
    image = cv2.cvtColor(self,cv2.COLOR_BGR2RGB)
    boxes = fr.face_locations(image)
    encoding = fr.face_encodings(image,boxes)
    d = [{"imagePath": i, "loc": box, "encoding": enc}
        for (box, enc) in zip(boxes, encoding)]

    return d

def cluster(self):
    path = 'cluster/'
    img_dir = os.listdir(path)
    for i in img_dir:
        shutil.rmtree(path + i)
    encoding = [d["encoding"] for d in self]
    clt = DBSCAN(eps=0.35,min_samples=3,metric="euclidean")
    clt.fit(encoding)
    label_ids = np.unique(clt.labels_)
    num_unique_faces = len(np.where(label_ids > -1)[0])

    print(num_unique_faces)
    print(collections.Counter(clt.labels_))
    for labelID in label_ids:
        print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]
        for i in idxs:
            image = cv2.imread('org2/' + self[i]['imagePath'])
            print(self[i]["imagePath"])
            move_image(image, i, labelID)


def pose(self,i):
    source = self
    result = model_pose(source, conf=0.3)
    result_keypoint = result[0].keypoints.xyn[0]

    left_dx = result[0].keypoints.xyn[0][13][0] - result[0].keypoints.xyn[0][11][0]
    left_dy = result[0].keypoints.xyn[0][13][1] - result[0].keypoints.xyn[0][11][1]
    left_rad = math.atan2(abs(left_dy), abs(left_dx))
    left_deg = left_rad * 180 / math.pi

    right_dx = result[0].keypoints.xyn[0][14][0] - result[0].keypoints.xyn[0][12][0]
    right_dy = result[0].keypoints.xyn[0][14][1] - result[0].keypoints.xyn[0][12][1]
    right_rad = math.atan2(abs(right_dy), abs(right_dx))
    right_deg = right_rad * 180 / math.pi

    if 5<left_deg<85 and 5<right_deg<85:
        destination = "pose/sitdown/"+i
        shutil.copyfile(source, destination)
        print("sitdown person")
    else:
        destination = "pose/standup/"+i
        shutil.copyfile(source, destination)
        print("standup person")


if __name__=='__main__':
    sec = 1
    img = []
    path = 'org/'
    path2 = 'org2/'
    count = 0
    while 1:
        time.sleep(sec)
        face()
        if keyboard.is_pressed('enter'):
            print('시스템 종료')
            break
        if glob.glob(os.path.join(path, '*.jpg')):
            img_dir = path
            img_dir = os.listdir(img_dir)
            for i in img_dir:
                result = embedding(path+i,i)
                img.extend(result)
                print(img)
                pose(path+i, i)
                shutil.move(path+i,path2+i)
                count +=1
                if count == 10:
                    cluster(img)
                    count = 0