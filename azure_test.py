from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import matplotlib.pyplot as plt
import json
import cv2

endpoint = "https://ai3870-3d-scan-detection.cognitiveservices.azure.com/"
training_key = "c1409e8f827f4ccc87c243ec5b2b728b"
prediction_key = "b02dbaf1c320441e853d0fedc5947560"

project_id = "f897dd5e-7177-4e51-ac77-a09137325c3f"
publish_iteration_name = "Face Detection"

base_image_url = "C:/Users/me1spw/Documents/Projects/AI3870 - AROPCQA/kinect-object-detection/"

#=============================== HELPER FUNCTIONS =============================
def append_predictions(result, confidence_level = 0.60):
    predictions = []
    for prediction in result.predictions:
        if prediction.probability > confidence_level:
            predictions.append(prediction)
            print(prediction.probability, prediction.tag_name)
            
    return predictions

# Adds border box and text on images based off predictions
# Custom Vision gives bounding box as noramlized coordinates 
# so they need to be computed to X, Y, Width, and Height that
# Open CV uses
def add_boxes_to_images(img, predictions):
    for pred in predictions:
        x = int(pred.bounding_box.left * img.shape[0])
        y = int(pred.bounding_box.top * img.shape[1])

        width = x + int(pred.bounding_box.width * img.shape[0])
        height = y + int(pred.bounding_box.height * img.shape[1])

        img = cv2.rectangle(img, (x, y), (width, height), (0, 0, 255), 2)
        img = cv2.putText(img, pred.tag_name, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA, False)

# Shows the image
def show_inline_img(img):
    inline_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 20))
    plt.axis('off')
    
    plt.imshow(inline_img)

#==============================================================================

train = CustomVisionTrainingClient(training_key, endpoint=endpoint)
predict = CustomVisionPredictionClient(prediction_key, endpoint=endpoint)
project = train.get_project(project_id)

with open("./test_image2.jpg", mode="rb") as image_data:
    results = predict.detect_image(project.id, publish_iteration_name, image_data)

print (len(results.predictions))
for prediction in results.predictions:
    print (prediction.probability)
predictions = append_predictions(results)

img = cv2.imread("./test_image2.jpg", cv2.IMREAD_COLOR)
add_boxes_to_images(img, predictions)
show_inline_img(img)
cv2.imshow("Test", img)
input("Press Enter to exit")