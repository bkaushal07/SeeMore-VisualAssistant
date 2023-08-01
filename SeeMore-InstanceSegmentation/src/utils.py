import random

import cv2
import torch
import numpy as np

from src import coco_names


# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names.COCO_INSTANCE_CATEGORY_NAMES), 3))


def get_segmentation(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the model
        outputs = model(image)

    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())

    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)

    # get the masks
    masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

    # discard masks for objects which are below threshold
    masks = masks[: thresholded_preds_count]

    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]

    # discard bounding boxes below threshold value
    boxes = boxes[: thresholded_preds_count]

    # get the classes labels
    labels = [coco_names.COCO_INSTANCE_CATEGORY_NAMES[i] for i in outputs[0]['labels']][: thresholded_preds_count]

    return masks, boxes, labels


def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6  # transparency for the segmentation map
    gamma = 0   # scalar added to each sum

    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)

        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color

        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)

        # convert the original PIL image into NumPy format
        image = np.array(image)

        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)

        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, thickness=2)

        # put the label text above the objects
        cv2.putText(
            image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=2, lineType=cv2.LINE_AA)
    
    return image


def get_depth(image, model, shape):
    with torch.no_grad():
        # forward pass of the image through the model
        prediction = model(image)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=shape,
            mode='bicubic',
            align_corners=False,
        ).squeeze().cpu()

    # Scale to the 0-1 range
    min_depth = prediction.min()
    max_depth = prediction.max()
    return (prediction - min_depth) / (max_depth - min_depth)


def get_object_depth(depth, masks, labels):
    object_depth = []

    # Get the depth for each object
    for i in range(len(masks)):
        # Assign median depth the object
        depth_map = torch.masked_select(depth, torch.from_numpy(masks[i]))
        object_depth.append((labels[i], torch.median(depth_map).item()))

    return object_depth