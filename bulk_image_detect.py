import csv
import os
import cv2
import numpy as np

input_dir = r"input_images"
output_dir = r"output_images"

yolo_cfg = r"RAK_model_cfg_dir\yolov3-custom.cfg"
yolo_weights = r"RAK_model_cfg_dir\yolov3-custom_final.weights"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def draw_predict(frame, left, top, right, bottom, class_id):

    if class_id == 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, "box", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif class_id == 1:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, "shower", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    elif class_id == 2:
        cv2.putText(frame, "basin", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 3:
        cv2.putText(frame, "door", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 4:
        cv2.putText(frame, "toiletsheet", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 5:
        cv2.putText(frame, "exhaust", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 6:
        cv2.putText(frame, "corner_shower", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 7:
        cv2.putText(frame, "staircase", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 8:
        cv2.putText(frame, "double_basin", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 9:
        cv2.putText(frame, "kitchen_sink", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 10:
        cv2.putText(frame, "cook_top", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif class_id == 11:
        cv2.putText(frame, "double_door", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    else:
        cv2.putText(frame, "bathtub", (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


def get_outputs_names(net):
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def sym_naming(csv_file_name, x_min, y_min, x_max, y_max, symbol):
    csvfile = open(csv_file_name, "a")
    writer = csv.writer(csvfile)
    if symbol == 0:
        writer.writerow([x_min, y_min, x_max, y_max, 'box'])
    elif symbol == 1:
        writer.writerow([x_min, y_min, x_max, y_max, 'shower'])
    elif symbol == 2:
        writer.writerow([x_min, y_min, x_max, y_max, 'basin'])
    elif symbol == 3:
        writer.writerow([x_min, y_min, x_max, y_max, 'door'])
    elif symbol == 4:
        writer.writerow([x_min, y_min, x_max, y_max, 'toiletsheet'])
    elif symbol == 5:
        writer.writerow([x_min, y_min, x_max, y_max, 'exhaust'])
    elif symbol == 6:
        writer.writerow([x_min, y_min, x_max, y_max, 'corner_shower'])
    elif symbol == 7:
        writer.writerow([x_min, y_min, x_max, y_max, 'staircase'])
    elif symbol == 8:
        writer.writerow([x_min, y_min, x_max, y_max, 'double_basin'])
    elif symbol == 9:
        writer.writerow([x_min, y_min, x_max, y_max, 'kitchen_sink'])
    elif symbol == 10:
        writer.writerow([x_min, y_min, x_max, y_max, 'cook_top'])
    elif symbol == 11:
        writer.writerow([x_min, y_min, x_max, y_max, 'Exhaust_fan'])
    elif symbol == 12:
        writer.writerow([x_min, y_min, x_max, y_max, 'double_door'])
    else:
        writer.writerow([x_min, y_min, x_max, y_max, 'bathtub'])


def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            # print("scores", scores)
            class_id = np.argmax(scores)
            # print("class_id",class_id)
            confidence = scores[class_id]
            # print("confidence",confidence)
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                # print("class_id", class_id)
                boxes.append([left, top, width, height, class_id])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # print(boxes)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        final_boxes.append(box)
        draw_predict(frame, left, top, left + width, top + height, box[4])

        xmin = left
        ymin = top
        xmax = left + width
        ymax = top + height
        symbol = box[4]

        csv_file = os.path.join(output_dir, filein).split(".")[0] + ".csv"
        if not os.path.isfile(csv_file):
            csvfile = open(csv_file, "w")
            writer = csv.writer(csvfile)
            writer.writerow(["xmin", "ymin", "xmax", "ymax", "symbol"])
        else:
            sym_naming(csv_file, xmin, ymin, xmax, ymax, symbol)

    #  print("final_boxes",final_boxes)
    return final_boxes


net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

for dir_, _, files in os.walk(input_dir):
    for filein in files:
        # print(filein)
        if filein.endswith('.csv'):
            continue
        image = cv2.imread(os.path.join(dir_, filein))

        blob = cv2.dnn.blobFromImage(image, 1 / 255, (416, 416), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(get_outputs_names(net))

        faces = post_process(image, outs, 0.5, 0.4)
        fileout = output_dir + os.sep + filein
        print("INFO : Saved at - ", fileout)
        cv2.imwrite(fileout, image)
