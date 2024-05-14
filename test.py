from tensorflow.keras.models import load_model
import cv2
import numpy as np

class_names = ['healthy','PD1']

lesion_Classifier = load_model("./parkinson first stage99.h5")

img_array=cv2.imread('nh.jpg')
img_array=cv2.resize(img_array,(512,512))

img_yuv_1 = cv2.cvtColor(img_array,cv2.COLOR_BGR2RGB)

img_yuv = cv2.cvtColor(img_yuv_1,cv2.COLOR_RGB2YUV)

y,u,v = cv2.split(img_yuv)



clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(5,5))
y = clahe.apply(y)

y = cv2.GaussianBlur(y,(3,3),1)

img_array_1 = cv2.merge((y,u,v))
img_array = cv2.cvtColor(img_array_1,cv2.COLOR_YUV2RGB)
test_im = img_array.reshape(-1,512,512,3)

#print(class_names[lesion_Classifier.predict_classes(test_im)[0]])
class_probabilities = lesion_Classifier.predict(test_im)
#print(class_probabilities)
# Extract the predicted class index
predicted_class_index = np.argmax(class_probabilities, axis=1)[0]

# Print the predicted class name
print("Predicted class:", class_names[predicted_class_index])
if class_names[predicted_class_index] == 'healthy':
    print("You do not have Parkinson's Disease")
else:
    print("You have Parkinson's Disease")