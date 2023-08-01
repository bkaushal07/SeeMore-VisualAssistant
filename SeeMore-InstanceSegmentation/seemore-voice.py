import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
import pyttsx3

from src import utils


def main():
    # initialize the segmentation model
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    mask_rcnn_transforms = weights.transforms()
    mask_rcnn_model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)

    # initialize the depth model
    midas_model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform

    # set the computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the models onto the computation device and set to eval mode
    mask_rcnn_model = mask_rcnn_model.to(device).eval()
    midas_model = midas_model.to(device).eval()

    # initialize the text-to-speech engine
    engine = pyttsx3.init()

    # open the video capture object
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

    while True:
        # read the frame from the video capture
        ret, frame = cap.read()

        # convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # perform segmentation
        input_batch = Image.fromarray(frame_rgb.copy())
        input_batch = mask_rcnn_transforms(input_batch)
        input_batch = input_batch.unsqueeze(0).to(device)
        masks, boxes, labels = utils.get_segmentation(input_batch, mask_rcnn_model, 0.965)

        # perform depth estimation
        input_batch = frame_rgb.copy()
        input_batch = midas_transforms(input_batch).to(device)
        depth = utils.get_depth(input_batch, midas_model, frame.shape[:2])

        # get the depth of each object
        object_depth = utils.get_object_depth(depth, masks, labels)
        object_depth = np.array(object_depth)  # Convert to NumPy array

        # Filter objects based on depth threshold (e.g., 0.5)
        threshold = 0.5
        close_objects = [obj for obj, depth in zip(labels, object_depth) if depth[0] > str(threshold)]
        print(close_objects)

        # convert close objects and labels to speech
        speech_output = "Close objects: " + ", ".join(close_objects)
        engine.say(speech_output)
        engine.runAndWait()

        # visualize the segmentation and depth map
        seg_final_output = utils.draw_segmentation_map(frame_rgb, masks, boxes, labels)
        depth_final_output = (depth.numpy() * 255).astype(np.uint8)

        # show the segmentation and depth maps
        cv2.imshow('Segmentation', seg_final_output)
        cv2.imshow('Depth Map', depth_final_output)

        # check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release the video capture object and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
