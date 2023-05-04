import cv2
import numpy as np
import torch
from model import Conv2d_cd, CDCN, CDCNpp
import argparse


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--detector_path', type=str, default='haarcascade_frontalface_alt2.xml', help='The path for the weight of face detector')
    parser.add_argument('--model_weights', type=str, default='CDCNpp_P4_290.pkl', help='The weight of the model')
    config = parser.parse_args()
    return config

def preprocess(f):
    f = cv2.resize(f, (256, 256))
    f = f[:, :, ::-1].transpose((2, 0, 1))
    f = f[np.newaxis, :, :, :]
    f = np.array(f)
    f = torch.from_numpy(f.astype(np.float)).float()
    f = (f - 127.5) / 128
    return f

def main(model, cap, face_cascade, device):
    while True:
        # Read image from camera
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect and mark faces
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5)
        f = None
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            f = img[y:y + h, x:x + w]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if f is not None:
            f = preprocess(f)
            f = f.to(device)
            map_x = model(f)
            # map_x = map_x.detach().numpy()
            # map_x = map_x.transpose(1, 2, 0)
            sum = torch.sum(map_x)
            if sum >= 250:
                print(sum.item(), " true")
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                name = 'Real'
            else:
                print(sum.item(), " false")
                color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                name = 'Spoof'
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # cv2.imshow("video", img)
            # cv2.waitKey(3500)
            del f

        cv2.imshow("video", img)
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()


if __name__ == '__main__':
    config = config_parser()
    face_cascade = cv2.CascadeClassifier(config.detector_path)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model = CDCNpp().to(device)
    model.load_state_dict(torch.load(config.model_weights, map_location=torch.device(device)))
    model.eval()
    cap = cv2.VideoCapture(0)

    main(model, cap, face_cascade, device)


