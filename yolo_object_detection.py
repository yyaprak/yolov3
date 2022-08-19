import cv2
import numpy as np
import glob
import random


def yolo_test():
    
    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_1000.weights", "yolov3_testing.cfg")
    
    # Name custom object
    classes = ["cercospora","koala","merkep"]
    
    
    # Images path
    #images_path = glob.glob(r"D:\Pysource\Youtube\2020\105) Train Yolo google cloud\dataset\*.jpg")
    #images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/yolo_dataset/*.jpeg")
    #images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/test_images/*.jpeg")
    #images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/plant_desases/*.jpeg")
    images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/plant_diseases_test/*.jpg")
    #images_path = glob.glob(r"/home/iyilmaz/Desktop/nesne_takibi/downloaded_images/plant_diseases_2/*.jpg")
    
    
    layer_names = net.getLayerNames()
    #output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    # Insert here the path of your images
    random.shuffle(images_path)
    # loop through all the images
    for img_path in images_path:
        # Loading image
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape
    
        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
        net.setInput(blob)
        outs = net.forward(output_layers)
    
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        temp=0
        for out in outs:
            #print("out :{0}{1}",out,temp)
            temp=temp+1
            
            for detection in out:
                scores = detection[5:]
                #print("scores :",scores)
                
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                #if confidence > 0.8:
                if confidence > 0.3:
                    # Object detected
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)
    
    
        cv2.imshow("Image", img)
        #key = cv2.waitKey(0)
        
        if cv2.waitKey(0) == ord('q'):
            break
    
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    yolo_test()
  
