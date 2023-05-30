import torch
from torch.nn import functional as F
from torchvision import transforms

import numpy as np
from collections import Counter
import random

# image shape to square

def pad_to_square(image):
    '''
    input: numpy image
    output: padded image, [x_pad, y_pad]
    '''

    c, h, w = image.shape
    if h == w:
        return image, [0, 0]

    add = abs(h - w) // 2
    if h < w:
        image = torch.concat([torch.zeros((c, add, w)), image, torch.zeros((c, add, w))], dim=1)
        return image, [0, add]
    else:
        image = torch.concat([torch.zeros((c, h, add)), image, torch.zeros((c, h, add))], dim=2)
        return image, [add, 0]


# augmentations & multi-scale

def horizontal_flip(image, label):
    image = transforms.functional.hflip(image)
    for l in label:
        l[1] = 1 - l[1]
    return image, label

def x_transition(image, label):
    transition = random.random() * 0.4 - 0.2
    for l in label:
        if l[1] + transition >= 1 or l[1] + transition < 0:
            return image, label

    c, h, w = image.shape
    empty = int(abs(w * transition))
    if empty > 0:
        if transition > 0:
            image = torch.concat([torch.zeros((c, h, empty)), image[:, :, :-empty]], dim=2)
        elif transition < 0:
            image = torch.concat([image[:, :, empty:], torch.zeros((c, h, empty))], dim=2)
        for l in label:
            l[1] += transition
    return image, label

def y_transition(image, label):
    transition = random.random() * 0.4 - 0.2
    for l in label:
        if l[2] + transition >= 1 or l[2] + transition < 0:
            return image, label

    c, h, w = image.shape
    empty = int(abs(h * transition))
    if empty > 0:
        if transition > 0:
            image = torch.concat([torch.zeros((c, empty, w)), image[:, :-empty, :]], dim=1)
        elif transition < 0:
            image = torch.concat([image[:, empty:, :], torch.zeros((c, empty, w))], dim=1)
        for l in label:
            l[2] += transition
    return image, label

def resize(image, size):
    return F.interpolate(image.unsqueeze(0), size, mode='bilinear', align_corners=True).squeeze(0)

# Calculate IOU of two boxes
def calculate_iou(pred, target):
    '''
    pred(tensor): x, y, w, h of a predicted box
    target(tensor): x, y, w, h of a ground truth box
    returns a float number of iou
    '''

    area_pred = pred[2] * pred[3]
    area_target = target[2] * target[3]
    area_intersect_width = max(0, min(pred[0] + pred[2] / 2, target[0] + target[2] / 2) - max(pred[0] - pred[2] / 2, target[0] - target[2] / 2))
    area_intersect_height = max(0, min(pred[1] + pred[3] / 2, target[1] + target[3] / 2) - max(pred[1] - pred[3] / 2, target[1] - target[3] / 2))
    area_intersect = area_intersect_width * area_intersect_height

    return area_intersect / (area_pred + area_target - area_intersect)

def calculate_ious(preds, target):
    '''
    preds(tensor): multi-dimensional boxes(x, y, w, h) => shape example: (7, 7, 3, 4)
    target(tensor): x, y, w, h of a ground truth box
    returns a tensor of iou values => shape example: (7, 7, 3)
    '''
    device = preds.device
    target = target.to(device)
    preds_shape = preds.shape
    preds = preds.reshape(-1, preds_shape[-1])
    n = preds.shape[0]

    area_preds = preds[:, 2] * preds[:, 3]
    area_target = target[2] * target[3]
    area_intersect_widths = torch.max(torch.concat([torch.tensor([0]).to(device).repeat(n, 1), (torch.min(torch.concat([(preds[:, 0] + preds[:, 2] / 2).reshape(-1, 1), (target[0] + target[2] / 2).repeat(n, 1)], dim=1), dim=1).values - torch.max(torch.concat([(preds[:, 0] - preds[:, 2] / 2).reshape(-1, 1), (target[0] - target[2] / 2).repeat(n, 1)], dim=1), dim=1).values).reshape(-1, 1)], dim=1), dim=1).values
    area_intersect_heights = torch.max(torch.concat([torch.tensor([0]).to(device).repeat(n, 1), (torch.min(torch.concat([(preds[:, 1] + preds[:, 3] / 2).reshape(-1, 1), (target[1] + target[3] / 2).repeat(n, 1)], dim=1), dim=1).values - torch.max(torch.concat([(preds[:, 1] - preds[:, 3] / 2).reshape(-1, 1), (target[1] - target[3] / 2).repeat(n, 1)], dim=1), dim=1).values).reshape(-1, 1)], dim=1), dim=1).values
    area_intersects = area_intersect_widths * area_intersect_heights

    return (area_intersects / (area_preds + area_target - area_intersects)).reshape(preds_shape[:-1])

# Define a loss funcion
def Yolo_loss(num_classes, lambda_coord=5, lambda_noobj=0.5):
    '''
    num_classes: number of classes
    lambda_coord: multiplied to coordinate losses
    lambda_noobj: multiplied to no_object losses

    returns a function which calculates the loss
        output: output of the model
        target: [[truth1, truth2,...for image1], [truth1, truth2...for image2]...]
    '''
    def loss_fn(output, target):
        n, _, d, d = output.shape
        device = output.device
        pred_class_prob = F.softmax(output[:, :num_classes].permute(0, 2, 3, 1), dim=3)
        pred_box = output[:, num_classes:].permute(0, 2, 3, 1).reshape(n, d, d, -1, 5)

        loss_xy = torch.tensor(0, device=device, dtype=torch.float32)
        loss_wh = torch.tensor(0, device=device, dtype=torch.float32)
        loss_conf = torch.tensor(0, device=device, dtype=torch.float32)
        loss_class = torch.tensor(0, device=device, dtype=torch.float32)

        for i in range(len(target)):
            visited = set([])
            conf_coord = []
            conf_obj = []
            target_box = torch.zeros((d, d, pred_box.shape[3]))
            for t in target[i]:
                row = int(t[2] / (1 / d))
                col = int(t[1] / (1 / d))
                if row >= d or row < 0 or col >= d or col < 0:
                    print(f'(Loss function) The target grid is out of range! row: {row}, col: {col}')
                if (row, col) in visited:
                    continue
                visited.add((row, col))

                xy_ingrid = t[1:3] * d - torch.tensor([col, row])
                wh_ingrid = t[3:] * d

                ious = calculate_ious(pred_box[i, row, col, :, 1:] * torch.tensor([1, 1, d, d]).to(device), torch.concat([xy_ingrid, wh_ingrid]))
                max_i = torch.argmax(ious)

                # calculate loss_xy, loss_wh of a target box of an image
                loss_xy += torch.sum(torch.square(pred_box[i, row, col, max_i, 1:3] - xy_ingrid.to(device))) * lambda_coord
                loss_wh += torch.sum(torch.square(
                    torch.sqrt(torch.abs(pred_box[i, row, col, max_i, 3:]) + 1e-9) \
                    - torch.sqrt(torch.abs(t[3:].to(device)) + 1e-9)
                )) * lambda_coord

                # calculate iou of a target box
                row_min = max(int((t[2] - t[4] / 2) / (1 / d)), 0)
                row_max = min(int((t[2] + t[4] / 2) / (1 / d)), d - 1)
                col_min = max(int((t[1] - t[3] / 2) / (1 / d)), 0)
                col_max = min(int((t[1] + t[3] / 2) / (1 / d)), d - 1)
                iou_matrix = calculate_ious(pred_box[i, row_min:row_max+1, col_min:col_max+1, :, 1:] * torch.tensor([1, 1, d, d]).to(device), torch.concat([xy_ingrid, wh_ingrid]))
                target_box[row_min:row_max+1, col_min:col_max+1, :] = iou_matrix * (lambda_noobj ** 0.5)
                conf_coord.append((row, col, max_i))
                conf_obj.append(iou_matrix[row - row_min, col - col_min, max_i])

                # calculate loss_class of a target box of an image
                target_class_prob = torch.zeros((num_classes))
                target_class_prob[int(t[0])] = torch.tensor(1.0)

                loss_class += torch.sum(torch.square(pred_class_prob[i, row, col] - target_class_prob.to(device)))

            # calculate loss_conf of target boxes of an image
            temp_box_conf = torch.sigmoid(pred_box[i, :, :, :, 0]) * (lambda_noobj ** 0.5)
            for j, (r, c, m) in enumerate(conf_coord):
                target_box[r, c, m] = conf_obj[j]
                temp_box_conf[r, c, m] = torch.sigmoid(pred_box[i, r, c, m, 0])
            loss_conf += torch.sum(torch.square(temp_box_conf - target_box.to(device)))

        # print(loss_xy.item(), loss_wh.item(), loss_conf.item(), loss_class.item())
        return (loss_xy + loss_wh + loss_conf + loss_class) / n

    return loss_fn

# Box reducing
def non_max_suppression(pred_per_classes, conf_th, nms_th):
    '''
    pred_per_classes: list of preds list by image [[pred1, pred2...for image1], [pred1, pred2...for image2]...]
        each pred is a list of predictions with index as classes.
    conf_th: boxes with confidence score under conf_th would be deleted.
    nms_th: iou larger than nms_th will be merged to a box of larger confidence score.
    '''

    n = len(pred_per_classes)
    num_classes = len(pred_per_classes[0])
    pred_nms_processed = [[[] for _ in range(num_classes)] for _ in range(n)]
    for i in range(n):
        for c in range(len(pred_per_classes[i])):
            pred_list = [x for x in pred_per_classes[i][c] \
                         if x[0] >= conf_th \
                            and 0 < x[1] < 1 \
                            and 0 < x[2] < 1 \
                            and 0 < x[3] < 1 \
                            and 0 < x[4] < 1]
            pred_list = sorted(pred_list, key=lambda x: x[0], reverse=True)
            while len(pred_list) > 0:
                temp_list = []
                pred_nms_processed[i][c].append(pred_list[0])
                for j in range(1, len(pred_list)):
                    if calculate_iou(pred_nms_processed[i][c][-1][1:], pred_list[j][1:]) < nms_th:
                        temp_list.append(pred_list[j])
                pred_list = sorted(temp_list, key=lambda x: x[0], reverse=True)

    return pred_nms_processed

# Pre-process for scoring output predictions
def pre_score_fn_Yolo(num_classes, use_nms=True):

    def pre_score_fn(output, prob_th=0, conf_th=0.5, nms_th=0.2):
        n, _, d, d = output.shape
        preds_temp = output[:, 4:].permute(0, 2, 3, 1).reshape(n, d, d, -1, 5)
        k = preds_temp.shape[3]
        add_xy = torch.tensor([[[0, i, j, 0, 0] for i in range(d)] for j in range(d)]).reshape(1, d, d, 1, 5)
        mul_xy = torch.tensor([1, d, d, 1, 1]).reshape(1, 1, 1, 1, 5)
        preds = ((preds_temp + add_xy) / mul_xy).reshape(n, -1, 5)
        preds[:, :, 3:] = torch.abs(preds_temp.reshape(n, -1, 5)[:, :, 3:])
        preds[:, :, 0] = torch.sigmoid(preds_temp.reshape(n, -1, 5)[:, :, 0])
        probs = torch.softmax(output[:, :4].permute(0, 2, 3, 1), dim=3).reshape(n, -1, 4)
        pred_per_classes = [[[] for _ in range(num_classes)] for _ in range(n)]
        for i in range(n):
            for j, prob in enumerate(probs[i]):
                if torch.max(prob) >= prob_th:
                    preds_classprob = preds[i][k * j: k * (j + 1)]
                    preds_classprob[:, 0] = torch.max(prob) * preds[i, 3 * j: 3 * (j + 1), 0]
                    pred_per_classes[i][torch.argmax(prob)].extend(preds_classprob)

        if use_nms:
            return non_max_suppression(pred_per_classes, conf_th, nms_th)
        return pred_per_classes

    return pre_score_fn

# Deciding True or False of a box with confidence
def average_precision(iou_th=0.5):
    '''
    iou_th: Threshold of iou for True positive

    returns a function which calculates the APs of an image
        preds: List of predictions with index as classes. len(preds) is equal to num_classes.
            The order of predictions is (class_confidence, x, y, w, h)
        target: List of ground truth of class and boxes.
            The order of ground truth is (class, x, y, w, h)
    '''

    def AP(preds, target):
        num_target_by_classes = Counter([int(t[0]) for t in target])

        preds_checked = {}
        for i, pred in enumerate(preds): # i: class
            if num_target_by_classes[i] == 0:
                continue
            # check if the prediction is true or false
            pred_sorted = sorted(pred, key=lambda x: x[0], reverse=True)
            duplication_check = set([])
            pred_checked = [False] * len(pred_sorted)
            for j, p in enumerate(pred_sorted):
                for k, t in enumerate(target):
                    if k in duplication_check:
                        continue
                    if t[0] == i and calculate_iou(p[1:], t[1:]) > iou_th:
                        pred_checked[j] = True

            preds_checked[i] = list(zip([p[0] for p in pred_sorted], pred_checked))

        return preds_checked

    return AP

# Calculate mAP from boxes
def mean_average_precision(pred_checked, num_target):
    '''
    pred_checked: hashmap with key of classes and value of list of T/F with confidence
    '''

    AP = []
    for key in pred_checked.keys():
        num_target_by_classes = num_target[key]

        if num_target_by_classes == 0:
            continue
        # calculate AP
        all_pred_sorted = sorted(pred_checked[key], key=lambda x: x[0], reverse=True)
        precision_recall_11 = []
        precision = [0, 0]
        recall = [0, num_target_by_classes]
        precision_calculated = 0
        recall_calculated = 0
        for _, p in all_pred_sorted:
            if p:
                precision[0] += 1
                precision[1] += 1
                recall[0] += 1
            else:
                precision[1] += 1
            precision_calculated = precision[0] / precision[1]
            recall_calculated = recall[0] / recall[1]
            if int(recall_calculated * 10) >= len(precision_recall_11):
                for _ in range(len(precision_recall_11), int(recall_calculated * 10) + 1):
                    precision_recall_11.append(precision_calculated)
            if recall_calculated == 1:
                break
        for _ in range(len(precision_recall_11), 11):
            precision_recall_11.append(0)

        AP.append(sum(precision_recall_11) / 11)

    if len(AP) == 0:
        return 0

    return sum(AP) / len(AP)

# Calculate mAP from model output
def score_fn_Yolo(preds_checker):
    '''
    preds: list of preds list by image [[pred1, pred2...for image1], [pred1, pred2...for image2]...]
        each pred is a list of predictions with index as classes.
    target: [[truth1, truth2,...for image1], [truth1, truth2...for image2]...]
    '''

    def score(preds, target):
        all_preds_checked = {}
        for i in range(len(preds)):
            preds_checked = preds_checker(preds[i], target[i])
            for key in preds_checked.keys():
                if key in all_preds_checked:
                    all_preds_checked[key].extend(preds_checked[key])
                else:
                    all_preds_checked[key] = preds_checked[key]

        num_target = Counter([])
        for i in range(len(target)):
            num_target += Counter([int(t[0].item()) for t in target[i]])

        return mean_average_precision(all_preds_checked, num_target)

    return score