import os
import cv2
import stat
import numpy as np
from mtcnn import MTCNN


__all__ = [
    'align_image', 'preprocess'
]


DETECTOR = MTCNN()


def rmtree(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.remove(filename)
        for name in dirs:
            filename = os.path.join(root, name)
            os.chmod(filename, stat.S_IWUSR)
            os.rmdir(filename)
    os.rmdir(top)


def rect_point_dist(bb, img_center):
    x, y, w, h = bb[0], bb[1], bb[2], bb[3]
    rect_center = (np.array([x, y]) + np.array([x + w, y + h])) / 2
    ry, rx = rect_center[0], rect_center[1]
    cy, cx = img_center[0], img_center[1]
    dx = max(np.abs(cx - rx) - w / 2, 0)
    dy = max(np.abs(cy - ry) - h / 2, 0)
    return dx * dx + dy * dy


def crop_image(img, margin=True):
    det = DETECTOR.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    nrof_faces = len(det)

    # only one face detected
    if nrof_faces == 1:
        det = det[0]

    # no faces detected, return whole image
    if nrof_faces == 0:
        print("No faces found in rotated image!")
        raise Exception("No faces found in rotated image!")

    # multiple faces detected, choose the centermost face
    if nrof_faces > 1:
        dists = []
        for d in det:
            x, y, w, h = d["box"][0], d["box"][1], d["box"][2], d["box"][3]
            bb = np.array([[x, y], [x + w, y + h]])
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
            '''
            img = cv2.rectangle(img, (bb[0, 0], bb[0, 1]), (bb[1, 0], bb[1, 1]), (0, 255, 0), 2)
            cv2.imwrite("tom_1.jpg", img)
            '''
            dists.append(rect_point_dist(d["box"], img_center))
        idx = np.argmin(dists)
        det = det[idx]

    x, y, w, h = det["box"][0], det["box"][1], det["box"][2], det["box"][3]
    bb = np.array([[x, y], [x + w, y + h]])
    kp = det["keypoints"]

    '''
    img = cv2.rectangle(img, (bb[0, 0], bb[0, 1]), (bb[1, 0], bb[1, 1]), (0, 255, 0), 2)
    cv2.imwrite("tom_4.jpg", img)
    '''
    if margin:
        y_margin = int(0.075 * ((y + h) + y) / 2.)
        x_margin = int(0.05 * ((x + w) + x) / 2.)
        y_t = max(y - y_margin, 0)
        y_b = min(y + h + y_margin, img.shape[0])
        x_t = max(x - x_margin, 0)
        x_b = min(x + w + x_margin, img.shape[1])
        crop_img = img[y_t:y_b, x_t:x_b]
    else:
        crop_img = img[y:y + h, x:x + w]

    '''
    crop_img = cv2.rectangle(crop_img, (bb[0, 0], bb[0, 1]), (bb[1, 0], bb[1, 1]), (0, 255, 0), 2)
    cv2.imwrite("tom_5.jpg", crop_img)
    '''

    return crop_img


def align_image(img, margin=True):
    det = DETECTOR.detect_faces(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    nrof_faces = len(det)

    # only one face detected
    if nrof_faces == 1:
        det = det[0]

    # no faces detected, return whole image
    if nrof_faces == 0:
        return img

    # multiple faces detected, choose the centermost face
    if nrof_faces > 1:
        dists = []
        for d in det:
            x, y, w, h = d["box"][0], d["box"][1], d["box"][2], d["box"][3]
            bb = np.array([[x, y], [x + w, y + h]])
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
            '''
            img = cv2.rectangle(img, (bb[0, 0], bb[0, 1]), (bb[1, 0], bb[1, 1]), (0, 255, 0), 2)
            cv2.imwrite("tom_1.jpg", img)
            '''
            dists.append(rect_point_dist(d["box"], img_center))
        idx = np.argmin(dists)
        det = det[idx]

    x, y, w, h = det["box"][0], det["box"][1], det["box"][2], det["box"][3]
    bb = np.array([[x, y], [x + w, y + h]])
    kp = det["keypoints"]

    # get angle between eyes
    dY = kp["right_eye"][1] - kp["left_eye"][1]
    dX = kp["right_eye"][0] - kp["left_eye"][0]
    angle = np.degrees(np.arctan2(dY, dX))

    # get center point between eyes
    eye_center = np.zeros((3,))
    eye_center[0] = (kp["left_eye"][0] + kp["right_eye"][0]) / 2
    eye_center[1] = (kp["left_eye"][1] + kp["right_eye"][1]) / 2
    eye_center[2] = 1.

    '''
    for k, p in kp.items():
        img = cv2.circle(img, (p[0], p[1]), 3, (0, 255, 0), 2)
    img = cv2.circle(img, (int(eye_center[0]), int(eye_center[1])), 3, (0, 0, 255), 2)
    cv2.imwrite("tom_2.jpg", img)
    '''

    # get center point of image
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # get size of image after rotation
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform rotation
    rot_img = cv2.warpAffine(img, M, (nW, nH))
    eye_center = np.dot(M, eye_center)

    '''
    for k, p in kp.items():
        p = np.dot(M, np.array([p[0], p[1], 1.]))
        rot_img = cv2.circle(rot_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), 2)
    rot_img = cv2.circle(rot_img, (int(eye_center[0]), int(eye_center[1])), 3, (0, 0, 255), 2)
    cv2.imwrite("tom_3.jpg", rot_img)
    '''

    try:
        img = crop_image(rot_img, margin)
    except Exception as e:
        pass
    return img


def preprocess(dataset_name):
    project_path = os.path.dirname(os.path.abspath(""))
    data_path = os.path.join(project_path, "data", dataset_name)
    dataset_path = os.path.join(project_path, "data", dataset_name + "_dataset")
    if os.path.exists(dataset_path):
        rmtree(dataset_path)

    os.mkdir(dataset_path)

    for subject in os.listdir(data_path):
        os.mkdir(os.path.join(dataset_path, subject))

        for image in os.listdir(os.path.join(data_path, subject)):
            print(os.path.join(data_path, subject, image))
            img = cv2.imread(os.path.join(data_path, subject, image))
            crop_img = align_image(img)
            '''
            cv2.imshow("image", img)
            cv2.imshow("aligned image", crop_img)
            cv2.waitKey(0)
            '''
            cv2.imwrite(os.path.join(dataset_path, subject, image), crop_img)


if __name__ == "__main__":
    preprocess("gtfd")
