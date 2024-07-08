import cv2
import time
import torch
import argparse
import numpy as np
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt, strip_optimizer
from utils.plots import output_to_keypoint, plot_one_box_kpt, colors
from sort import Sort

@torch.no_grad()
def run(poseweights="yolov7-w6-pose.pt", source="football1.mp4", device='cpu', view_img=False,
        save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
    
    frame_count = 0
    total_fps = 0
    time_list = []
    fps_list = []
    
    device = select_device(device)
    half = device.type != 'cpu'

    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()
    names = model.module.names if hasattr(model, 'module') else model.names

    # Initialize SORT tracker
    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)
    
    if source.isnumeric():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
   
    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        raise SystemExit()

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    vid_write_image = letterbox(cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out = cv2.VideoWriter(f"{source}_keypoint.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), 30,
                          (resize_width, resize_height))

    trajectories = {}

    while cap.isOpened():
        print(f"Frame {frame_count + 1} Processing")

        ret, frame = cap.read()
        
        if ret:
            orig_image = frame
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))
            
            image = image.to(device)
            image = image.float()
            start_time = time.time()
            
            with torch.no_grad():
                output_data, _ = model(image)

            output_data = non_max_suppression_kpt(output_data,
                                                  0.25,
                                                  0.65,
                                                  nc=model.yaml['nc'],
                                                  nkpt=model.yaml['nkpt'],
                                                  kpt_label=True)
            
            output = output_to_keypoint(output_data)

            im0 = image[0].permute(1, 2, 0) * 255
            im0 = im0.cpu().numpy().astype(np.uint8)
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            bbox_data = []

            for i, pose in enumerate(output_data):
                if len(pose):
                    for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                        if int(cls) == 0:  # Filter for person class (class 0)
                            kpts = pose[det_index, 6:]
                            label = None if hide_labels else (names[int(cls)] if hide_conf else f'{names[int(cls)]} {conf:.2f}')
                            plot_one_box_kpt(xyxy, im0, label=label, color=colors(int(cls), True),
                                             line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                             orig_shape=im0.shape[:2])
                            bbox_data.append([*xyxy, conf, cls])

            if bbox_data:
                bbox_data = np.array(bbox_data)
                tracked_objects = sort_tracker.update(bbox_data)

                # Print the number of detected humans
                print(f"Number of humans detected: {len(tracked_objects)}")

                for obj in tracked_objects:
                    bbox = obj[:4].astype(int)
                    track_id = int(obj[4])
                    cls = int(obj[5])
                    label = f'ID: {track_id}'

                    # Store the trajectory
                    if track_id not in trajectories:
                        trajectories[track_id] = []
                    center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
                    trajectories[track_id].append(center)

                    plot_one_box_kpt(bbox, im0, label=label, color=colors(cls, True),
                                     line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                     orig_shape=im0.shape[:2])
                    
                    # Draw trajectory
                    for j in range(1, len(trajectories[track_id])):
                        if trajectories[track_id][j - 1] is None or trajectories[track_id][j] is None:
                            continue
                        cv2.line(im0, trajectories[track_id][j - 1], trajectories[track_id][j], (0, 255, 0), 2)
                
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1

            fps_list.append(total_fps)
            time_list.append(end_time - start_time)
            
            if view_img:
                cv2.imshow("YOLOv7 Pose Estimation and Tracking", im0)
                if cv2.waitKey(1) == ord('q'):
                    break

            out.write(im0)
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    plot_fps_time_comparision(time_list=time_list, fps_list=fps_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='football1.mp4', help='video/0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    opt = parser.parse_args()
    return opt


def plot_fps_time_comparision(time_list, fps_list):
    plt.figure()
    plt.xlabel('Time (s)')
    plt.ylabel('FPS')
    plt.title('FPS and Time Comparison Graph')
    plt.plot(time_list, fps_list, 'b', label="FPS & Time")
    plt.savefig("FPS_and_Time_Comparison_pose_estimate.png")


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)
