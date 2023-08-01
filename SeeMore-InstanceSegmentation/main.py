import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

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

    # load the model on to the computation device and set to eval mode
    mask_rcnn_model = mask_rcnn_model.to(device).eval()
    midas_model = midas_model.to(device).eval()

    # TODO: modify from here for webcam video processing
    # just put the remaining of the codebase from here, within
    # the loop for the video capture object
    image_path = 'input.jpg'
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    # perform segmentation
    input_batch = Image.fromarray(orig_image.copy())
    input_batch = mask_rcnn_transforms(input_batch)
    input_batch = input_batch.unsqueeze(0).to(device)
    masks, boxes, labels = utils.get_segmentation(input_batch, mask_rcnn_model, 0.965)

    # perform depth estimation
    input_batch = orig_image.copy()
    input_batch = midas_transforms(input_batch).to(device)
    depth = utils.get_depth(input_batch, midas_model, orig_image.shape[: 2])

    # get the depth of each object
    object_depth = utils.get_object_depth(depth, masks, labels)

    # TODO: this will give you the list of each object of their disparity
    # note that closest object is 1 and farthest object is 0. It'll be in 0-1 range
    # So put some threshold like, greater than 0.8 means its close
    # idk what value that will be, depends on your webcam and your testing
    print(object_depth)

    # set the save path
    depth_final_output = (depth.numpy() * 255).astype(np.uint8)
    seg_final_output = utils.draw_segmentation_map(orig_image, masks, boxes, labels)
    cv2.imwrite('seg.png', seg_final_output)
    cv2.imwrite('depth.png', depth_final_output)


if __name__ == '__main__':
    main()