# People-detection-using-yolo

This program uses yolo(you only look once) model to selectively recognize people using a live camera feed, now commonly yolo detect more than 80 objects in a frame but you can play around the code and coco.name(label) file to filter out the required objects you want to detect, for human body you can just use my code.

## Yolo model- 
You can use two type of weights and cfg file to get different result
1. Tiny weights and cgf which will provide better frame rate but low accuracy

![Tiny](https://github.com/Micky659/People-detection-using-yolo/blob/master/Output/outputTiny.gif)


2. Normal weights and cgf which will be highly accurate but compensate in frame rate

![Normal](https://github.com/Micky659/People-detection-using-yolo/blob/master/Output/outputNormal.gif)

***You can download weights and cfg file as per your requirement from this website [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/), after downloading put the weight in weight folder and cfg file in cfg folder and change the code at line number 7.***

## Requirments-
1. Python 3.2 or more [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Open CV *pip install opencv-python*
3. Numpy *pip install numpy*.


**Feel free to build over my code and use it wisely**


>stock Video by Creativ Medium from Pexels
