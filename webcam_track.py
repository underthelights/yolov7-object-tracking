import cv2
import time
import torch
import argparse
import numpy as np
from numpy import random
from torchvision import transforms
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import output_to_keypoint
from utils.torch_utils import select_device
from sort import Sort

class PoseEstimator:
    def __init__(self, opt):
        self.opt = opt
        self.device = select_device(opt.device)
        self.model = attempt_load(opt.poseweights, map_location=self.device)
        self.model.eval()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        self.frame_count = 0
        self.total_fps = 0

        # Initialize SORT tracker
        self.sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    def process_image(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (self.opt.img_size), stride=64, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        image = image.to(self.device)
        image = image.float()
        start_time = time.time()

        with torch.no_grad():
            output_data, _ = self.model(image)

        output_data = non_max_suppression_kpt(output_data,
                                              0.25,
                                              0.65,
                                              nc=self.model.yaml['nc'],
                                              nkpt=self.model.yaml['nkpt'],
                                              kpt_label=True)

        output = output_to_keypoint(output_data)

        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

        pose_data = []
        bbox_data = []

        for pose in output_data:
            if len(pose):
                for det_index, (*xyxy, conf, cls) in enumerate(pose[:, :6]):
                    kpts = pose[det_index, 6:]
                    person_data = kpts.tolist()
                    pose_data.append(person_data)
                    bbox_data.append([*xyxy, conf, cls])  # Add class info here

        # Update SORT tracker
        tracked_objects = self.sort_tracker.update(np.array(bbox_data))

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        self.total_fps += fps
        self.frame_count += 1

        if self.opt.view_img:
            cv2.putText(im0, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            for obj in tracked_objects:
                bbox = obj[:4].astype(int)
                track_id = int(obj[4])
                cv2.rectangle(im0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(im0, f'ID: {track_id}', (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.imshow("YOLOv7 Pose Estimation and Tracking", im0)
            cv2.waitKey(1)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    opt = parser.parse_args()
    return opt

def main(opt):
    pose_estimator = PoseEstimator(opt)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_estimator.process_image(frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
