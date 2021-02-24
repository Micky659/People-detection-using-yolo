import cv2
import numpy as np


def load_yolo():
    # Change weights and cgf file below to get different outputs
    net = cv2.dnn.readNet("Weights/yolov3.weights", "Cfg/yolov3.cfg")
    with open("Labels/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    return net, colors, output_layers

def start_webcam():
    # For webcam feed put the number assigned to your camera(default is 0)
    # For video feed put the path of your video with extension
    cap = cv2.VideoCapture('Stock/people.mp4')

    return cap


def display_blob(blob):
    '''
        Three images each for RED, GREEN, BLUE channel
    '''
    for b in blob:
        for n, imgb in enumerate(b):
            cv2.imshow(str(n), imgb)


def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs


def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[0]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
    return boxes, confs


def draw_boxes(boxes, confs, colors, img):
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    humans = 1
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
            humans += 1
    cv2.putText(img, f'Total Humans : {humans-1}', (60, 80), font, 5, (255, 0, 255), 6)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # You can manually tweak the dimensions of the output screen according to your video or webcam feed
    cv2.resizeWindow('image', 696, 432)
    cv2.imshow('image', img)

def webcam_detect():
    model, colors, output_layers = load_yolo()
    cap = start_webcam()
    while True:
        _, frame = cap.read()
        height, width, channels = frame.shape
        blob, outputs = detect_objects(frame, model, output_layers)
        boxes, confs = get_box_dimensions(outputs, height, width)
        draw_boxes(boxes, confs, colors, frame)
        # Press esc key to end webcam
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()


if __name__ == '__main__':
    # print("Select either option 1 or option 2 from below")
    # print("1. Use yolo tiny model-- Will give better frame rate at the cost of accuracy")
    # print("2. Use yolo model-- Will give extremely high accuracy but will compensate in frame rate")
    print('---- Detecting Humans ----')
    webcam_detect()
    cv2.destroyAllWindows()