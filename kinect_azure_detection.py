# Named pipe
import win32file

import numpy as np
import os, os.path
import tkinter as tk

# For visualization
import cv2
from PIL import ImageTk, Image
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
        
        return img

#------------------------------------------------------------------------------
def start_stream():

    # Create pipe client
    fileHandle = win32file.CreateFile("\\\\.\\pipe\\mynamedpipe",
        win32file.GENERIC_READ | win32file.GENERIC_WRITE,
        0, None,
        win32file.OPEN_EXISTING,
        0, None)

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

    return detections

#------------------------------------------------------------------------------
def save_depth_detections(detections, depth_img_full):

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

        # # Change 1D to xyz
        xyz = []
        for row in range(0, FRAME_HEIGHT):
            for col in range(0, FRAME_WIDTH):
                xyz.append([col, row, depth_img_full[row][col]])
        xyz = np.array(xyz)

        # PLY
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # o3d.io.write_point_cloud("/test.ply", pcd)

        # Increment index of detection
        index += 1       

#------------------------------------------------------------------------------
def detect_click():
    global pressed
    pressed = True

#------------------------------------------------------------------------------
def quit():
    global fileHandle
    end_stream(fileHandle)
    input("Press Enter to exit")
    gui.quit()

#=================================== MAIN =====================================
if __name__ == "__main__":

    # Setup UI
    gui = tk.Tk()
    gui.title("Kinect Azure Detection")
    gui.config(bg="white")

    stream_label = tk.Label(gui, bg="#EEEEEE")
    stream_label.grid(row=0, column=0)

    # detect_label = tk.Label(gui, bg="#EEEEEE")
    # detect_label.grid(row=0, column=1)

    detect_btn = tk.Button(gui, text="Detect", padx=10, pady=5, fg="white", bg="#0099DF", command=detect_click)
    detect_btn.grid(row=1, column=0, columnspan=2)

    # Begin..
    fileHandle = start_stream()

    while True:
        # Get latest sensor data
        depth_img_full, ab_img_full = stream(fileHandle)
        
        # Conver image format (numpy -> PIL -> ImageTK)      
        img = Image.fromarray(ab_img_full)
        tk_img = ImageTk.PhotoImage(img)

        # Display frame
        stream_label.grid_forget()
        stream_label = tk.Label(gui, image=tk_img)
        stream_label.grid(row=0, column=0)

        # On button press..
        if pressed:
            pressed = False

            # Call Azure to get detections
            detections = detect(ab_img_full)
            save_depth_detections(detections, depth_img_full)

            # Add bounding boxes and convert format (numpy -> PIL -> ImageTK)
            img = add_boxes_to_images(ab_img_full, detections)
            img_out = Image.fromarray(img)
            tk_img = ImageTk.PhotoImage(img_out)

            # Display frame
            stream_label.grid_forget()
            stream_label = tk.Label(gui, image=tk_img)
            stream_label.grid(row=0, column=0)
            
            gui.update()
            cv2.waitKey(2000)
            

        gui.update()