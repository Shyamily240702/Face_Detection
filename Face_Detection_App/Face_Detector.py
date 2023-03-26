import cv2 
from random import randrange


# Load some pre-trained data on face frontals from opencv (haarcascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # trained_face_data is owr own variable , CascadeClassifier is a function inside the cv2 which is trained to classify the faces based on the cascade algo.


#<<<!!!!! TO DETECT FACES FROM WEBCAM (real-time Video)  !!!!>>>

#To capture video from webcam
webcam = cv2.VideoCapture(0) # videocapture will capture the video (0) here so it's gonna get your default webcam capture your video from webcam
#webcam = cv2.VideoCapture('Children.mp4')#  <<!!to detect faces in a given video

#Iterate forever over frames
while True: #looping because a video is a collection frames not only a single frame 
     #Read the current frame
     successful_frame_read, frame = webcam.read()

     # Must convert to greyscale
     grayscaled_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

     # Detect faces
     face_coordinates = trained_face_data.detectMultiScale(grayscaled_video)

     #Draw Rectangles around the faces
     for(x,y,w,h) in face_coordinates:# to detect all the face in the photo we're looping 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 3)# 128 for light colours,randrange(256) will give random coloured rectangles 256 because it will take n-1 so to obtain 255 we give 256

     cv2.imshow('Face Detector', frame)
     key = cv2.waitKey(1) #if we don't have waitkey it won't display anything and !!! in img we gave waitkey() nothing inside() but here we are giving waitkey(1) this means the waitkey is functioned like if there is nothing inside() it will wait until a key is pressed before moving to the next frame, video is a collection of frames(images basically) so if we give waitkey(1) it will wait for 1 millisecond and then it will change the frame no need to press any key
     
     # Stop if Q key is pressed Q-quit ASCII OF Q-81 and ASCII of q-113
     if key == 81 or key == 113:
        break

# Release the VideoCapture object
webcam.release()

print("Code Completed")















# <<!!!!! TO DETECT FACES IN AN IMAGE !!!!!>>
"""
#<choose an image to detect faces in
img = cv2.imread('Family.jpg') # img is owr own variable, imread is a cv2 function which will read the image in 2D arrays here RDJ.png is the image we are giving. 

# Must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #cvtColor converts an image from one space to another here BGR is RGB in opencv RGB is always backwards BGR and it will convert to GRAY so BGR2GRAY

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) #Detects objects of different sizes in the input image. The detected objects are returned as a list of coordinates of rectangles.

#Draw Rectangles around the faces
for(x,y,w,h) in face_coordinates:# to detect all the face in the photo we're looping 
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(128,256),randrange(128,256),randrange(128,256)), 3)# 128 for light colours,randrange(256) will give random coloured rectangles 256 because it will take n-1 so to obtain 255 we give 256
#cv2.rectangle(img, (401,108), (401+336, 108+336), (0, 255, 0), 2) #here (401,108) are top left and bottom right rect. coordinate points  of the face respectively which is (x,y) and (401+336, 108+336) is (x+w,y+h) where w and h are width and height of the rect.  and (0,255,0) is bgr here it is green and the last 2 is the thickness of the rectangle
  #(x,y,w,h) = face_coordinates[0] # 0-large face 1-slight small face 2-small face # (x,y,w,h) is a tuple which is assigned the values from the face_coordinates
  #cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)# the rectangle is only green


#To print the face coordinates
#print(face_coordinates) # we'll get output something like [[c1 c2 c3 c4]] coordinates where c1 is the coordinate of the face on the top left corner same as c4 is the coordinate of the face at the bottom right corner


#To display the image with the faces spotted
cv2.imshow('Face Detector',img) # imshow is a cv2 function that will show the image, Face Detector is the name of the pop up of the image, img is the image which at line no.7.

#wait here in the code and listen for a key press
cv2.waitKey() #waitkey is needed because otherwise the image pop up will closes instantly

#print("Code Completed")

"""
#<!!! END OF DETECTING FACES FROM AN IMAGE !!!>

