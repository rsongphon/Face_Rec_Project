import face_recognition
import cv2
import os
import pickle


# Folder Structure : Know_faces/(name of person)
# inside (name of person) is a picture of know faces of this person
CWD_DIR = os.getcwd()
KNOWN_FACE_DIR = os.path.join(CWD_DIR,'known_faces')
UNKNOW_FACE_DIR = os.path.join(CWD_DIR,'unknown_faces')

#OpenCV Properties of rectangle to label the face
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_LINE = cv2.LINE_AA
FRAME_THICKNESS = 2
FONT_THICKNESS = 2

# Machice learning model
MODEL = 'cnn'

# encode face pickle file #

ENCODE_FILENAME = 'encode_faces.pkl'

# More tolerance = more detection but chance for error (False Positive)
TOLERANCE = 0.6

CAPTURE = cv2.VideoCapture(0)

# Returns (R, G, B) from name
def name_to_color(name):
    # Take 3 first letters, tolower()
    # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
    color = [(ord(c.lower())-97)*8 for c in name[:3]]
    return color

def rescaleFrame(frame,scale): 
    # work for image , video , live camera
    #The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color):
    width = int(frame.shape[1] * scale)  # shape[1] = width: must be int
    height = int(frame.shape[0]* scale) # shape[0] = :height must be int

    dimension = (width,height) # tuple to keep dimension

    return cv2.resize(frame,dimension,interpolation=cv2.INTER_AREA)

print('Loading face encode file....')
# load enconding faces # 
with open(ENCODE_FILENAME,'rb') as pickle_file:
    encoding_data = pickle.load(pickle_file)

# Loading unknow face
print('Webcam on ....')

while True:

    #Load an image 
    _,image = CAPTURE.read()

    # unknow image may have many face in the frame so we need to idetify location
    # also use location for labeling
    location_face = face_recognition.face_locations(image,model=MODEL)

    # Encoding face base on the location in the image
    # Without that it will search for faces once again slowing down whole process
    encoding_face = face_recognition.face_encodings(image,location_face)

    # there might be more faces in an image - we can find faces of dirrerent people
    print('Found {} faces'.format(len(encoding_face)))

    for face_encoding , face_location in zip(encoding_face,location_face):
        # for each face found in picture return array of boolean of size know face array we create before
        for person_name in encoding_data:
            results = face_recognition.compare_faces(encoding_data[person_name],face_encoding,TOLERANCE)
            print(results)

            #We can use result array to find index of know_face and print it name from know_name array (remember that it will be match)
            # grab only index that have True boolean value
            if True in results: # If at least one is true, get a name of first of found labels
                print(person_name)
                print(face_location)

                # Grab coordiation of the face location 
                # face_location contains positions in order: top, right, bottom, left
                # grab (x,y) coordination for top-left and buttom- right
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1],face_location[2])

                # Get color by name using function
                color = name_to_color(person_name)

                # Draw rectangle 
                cv2.rectangle(image,top_left,bottom_right,color,FRAME_THICKNESS)

                # Another regtangle to lable the name , below the frame
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1],face_location[2]+22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, person_name, (face_location[3] + 10, face_location[2] + 15), FONT, 0.5, (200, 200, 200), FONT_THICKNESS)

    # Show image
    cv2.imshow('Frame', image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

CAPTURE.release()
cv2.destroyAllWindows()