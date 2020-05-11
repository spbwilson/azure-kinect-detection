# Named pipe
import win32file

import numpy as np
import os, os.path
import tkinter as tk

# For visualization
import cv2
import matplotlib.pyplot as plt

# For PLY work
import open3d as o3d

# For Azure usage
from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient

#==============================================================================
#----- AZURE VARIABLES -----
endpoint = "https://ai3870-3d-scan-detection.cognitiveservices.azure.com/"
training_key = "c1409e8f827f4ccc87c243ec5b2b728b"
prediction_key = "b02dbaf1c320441e853d0fedc5947560"

project_id = "f897dd5e-7177-4e51-ac77-a09137325c3f"
publish_iteration_name = "Face Detection"

train = CustomVisionTrainingClient(training_key, endpoint=endpoint)
predict = CustomVisionPredictionClient(prediction_key, endpoint=endpoint)
project = train.get_project(project_id)

confidence = 0.5

#----- OS VARIABLES -----
base_image_url = os.getcwd()
img_out_name = base_image_url + "/frame_out.jpg"

#----- KINECT VARIABLES -----
# The image size of depth/ir (Assuming depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED, change it otherwise)
FRAME_WIDTH = 640
FRAME_HEIGHT = 576
BYTES_PER_PIXEL = 2

# For gray visulization
MAX_DEPTH_FOR_VIS = 8000.0
MAX_AB_FOR_VIS = 512.0

#----- UI -----
pressed = False


#==============================================================================

def append_predictions(result, confidence_level = 0.70):

    predictions = []
    for prediction in result.predictions:
        if prediction.probability > confidence_level:
            predictions.append(prediction)
            print(prediction.probability, prediction.tag_name)
            
    return predictions

#------------------------------------------------------------------------------
# Adds border box and text on images based off predictions
# Custom Vision gives bounding box as noramlized coordinates 
# so they need to be computed to X, Y, Width, and Height that
# Open CV uses
def add_boxes_to_images(img, predictions):

    for pred in predictions:
        # img.shape is row, column
        x = int(pred.bounding_box.left * img.shape[1])
        y = int(pred.bounding_box.top * img.shape[0])

        width = x + int(pred.bounding_box.width * img.shape[1])
        height = y + int(pred.bounding_box.height * img.shape[0])

        img = cv2.rectangle(img, (x, y), (width, height), (0, 0, 255), 2)
        img = cv2.putText(img, pred.tag_name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA, False)

#------------------------------------------------------------------------------
# Shows the image
def show_inline_img(img):

    inline_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 20))
    plt.axis('off')
    
    plt.imshow(inline_img)

#------------------------------------------------------------------------------
def start_stream():

    # Create pipe client
    fileHandle = win32file.CreateFile("\\\\.\\pipe\\mynamedpipe",
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None,
        win32file.OPEN_EXISTING,
        0, None)
    
    # For visualization
    cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('vis', FRAME_WIDTH * 2, FRAME_HEIGHT)

    return fileHandle

#------------------------------------------------------------------------------
def end_stream(fileHandle):

    win32file.CloseHandle(fileHandle)

#------------------------------------------------------------------------------
def stream(fileHandle):
    
    # Send request to pipe server
    request_msg = "Request depth image and ir image"
    win32file.WriteFile(fileHandle, request_msg.encode())
    
    # Read reply data, need to be in same order/size as how you write them in the pipe server
    # in pipe_streaming_example/main.cpp
    depth_data = win32file.ReadFile(fileHandle, FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL)
    ab_data = win32file.ReadFile(fileHandle, FRAME_WIDTH * FRAME_HEIGHT * BYTES_PER_PIXEL)
    
    # Reshape for image visualization
    depth_img_full = np.frombuffer(depth_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
    ab_img_full = np.frombuffer(ab_data[1], dtype=np.uint16).reshape(FRAME_HEIGHT, FRAME_WIDTH).copy()
    
    depth_vis = (plt.get_cmap("gray")(depth_img_full / MAX_DEPTH_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
    ab_vis = (plt.get_cmap("gray")(ab_img_full / MAX_AB_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
    
    # Visualize the images
    vis = np.hstack([depth_vis, ab_vis])
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    cv2.imshow("vis", vis)

    return depth_img_full, ab_img_full

#------------------------------------------------------------------------------
def detect(ab_img_full):
    
    ab_vis = (plt.get_cmap("gray")(ab_img_full / MAX_AB_FOR_VIS)[..., :3]*255.0).astype(np.uint8)

    # Use Azure service to detect objects in frame
    cv2.imwrite(img_out_name, ab_vis)
    with open(img_out_name, mode="rb") as image_data:
        results = predict.detect_image(project.id, publish_iteration_name, image_data)

    # Get predictions over confidence value
    detections = append_predictions(results, confidence)

    # Display detections
    img = cv2.imread(img_out_name, cv2.IMREAD_COLOR)
    add_boxes_to_images(img, detections)
    show_inline_img(img)
    cv2.imshow("Test", img)

    return detections

#------------------------------------------------------------------------------
def extract_3d(detections, depth_img_full):

    # Crop depth image
    depth_vis = (plt.get_cmap("gray")(depth_img_full / MAX_DEPTH_FOR_VIS)[..., :3]*255.0).astype(np.uint8)
    img = cv2.imread(img_out_name, cv2.IMREAD_COLOR)
    
    index = 0
    for det in detections:
        x = int(det.bounding_box.left * img.shape[1])
        y = int(det.bounding_box.top * img.shape[0])

        width = x + int(det.bounding_box.width * img.shape[1])
        height = y + int(det.bounding_box.height * img.shape[0])
        
        depth_out = depth_vis[y:y+height , x:x+width]

        # Save as depth image
        depth_out_name = base_image_url + "/detection_" + str(index) + ".jpg"
        cv2.imwrite(depth_out_name, depth_out)

        # Increment index of detection
        index += 1

        # # Change 1D to xyz
        # xyz = []
        # for row in range(0, FRAME_HEIGHT):
        #     for col in range(0, FRAME_WIDTH):
        #         xyz.append([col, row, depth_img_full[row][col]])
        # xyz = np.array(xyz)

        # # PLY
        # np_points = np.random.rand(100, 3)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(depth_img_full)
        # o3d.io.write_point_cloud("/test.ply", pcd) 
        
        #o3d_img = o3d.io.read_image(depth_out_name)
        #pcd = o3d.geometry.PointCloud.create_from_depth_image(extrinsic = depth_img_full)       

#------------------------------------------------------------------------------
def detect_click():
    print("Click!")
    global pressed 
    pressed = True

#=================================== MAIN =====================================
if __name__ == "__main__":

    # Setup UI
    root = tk.Tk()

    canvas = tk.Canvas(root, height=700, width=700, bg="white")
    canvas.pack()

    frame = tk.Frame(root, bg="#EEEEEE")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    detect_btn = tk.Button(root, text="Detect", padx=10, pady=5, fg="white", bg="#0099DF", command=detect_click)
    detect_btn.pack()

    # Begin..
    fileHandle = start_stream()

    while True:
        root.update()
        depth_img_full, ab_img_full = stream(fileHandle)
        
        if pressed:
            detections = detect(ab_img_full)
            extract_3d(detections, depth_img_full)
            pressed = False

        key = cv2.waitKey(1)
        if key == 27: # Esc key to stop
            break
    
    

    end_stream(fileHandle)
    input("Press Enter to exit")
    root.quit()