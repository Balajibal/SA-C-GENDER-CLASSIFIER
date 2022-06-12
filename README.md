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
```
/*
```
1. CODE :
![SKILL ASSESSMENT OUTPUT](XXX.png)
![Screenshot (210)](https://user-images.githubusercontent.com/75234946/172541086-7f86e8ea-3b04-4299-80d2-0e29204a27c8.png)
![Screenshot (211)](https://user-images.githubusercontent.com/75234946/172541130-329f6e49-049c-4472-b86e-b06ace4e1d4b.png)
![Screenshot (212)](https://user-images.githubusercontent.com/75234946/172541187-2e602b5f-9218-422b-8f7e-344fbb931db5.png)
![Screenshot (208)](https://user-images.githubusercontent.com/75234946/172541531-edff47aa-1de2-436a-820d-39191ba6f007.png)
![Screenshot (209)](https://user-images.githubusercontent.com/75234946/172541579-77df8722-150e-4c99-bbfd-6d635da04c32.png)
2. DEMO VIDEO YOUTUBE LINK:
3.```https://youtu.be/skINoVJB88U```

*/
```
