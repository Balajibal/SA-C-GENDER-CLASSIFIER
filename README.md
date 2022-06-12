# SA-C-GENDER-CLASSIFIER
# Algorithm
1.Import the required packages

2.Install deepFace

3.Read the image

4.Select age,gender,emotion as attribute

5.Face is analyzed and output is shown

## Program:
```python
/*
Program to implement Age,Emotion,Gender Classification
Developed by   : Balaji N
RegisterNumber :  212220230006
*/
```python
from deepface import DeepFace
import cv2 
import matplotlib.pyplot as plt
img=cv2.imread('WhatsApp Image 2022-06-07 at 2.39.12 PM (3).jpeg')
plt.imshow(img[:,:,::-1])
result = DeepFace.analyze(img,actions=['gender'])
result1 = DeepFace.analyze(img,actions=['age'])
result2 = DeepFace.analyze(img,actions=['emotion'])
print("Gender : ",result['gender'])
print("Age : ",result1['age'])
print("Emotion : ",result2['emotion'])
```

## OUTPUT:

1. CODE :

![Screenshot (208)](https://user-images.githubusercontent.com/75234946/173219567-f3b2f336-b812-4f56-98da-ef5319eb39fd.png)

![Screenshot (214)](https://user-images.githubusercontent.com/75234946/173219605-270b3dab-886b-49d4-831e-0539c01ec0fd.png)


2. DEMO VIDEO YOUTUBE LINK:
</br>
   https://youtu.be/skINoVJB88U
