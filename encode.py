import face_recognition
import pickle
import cv2
import os


# Folder Structure : Know_faces/(name of person)
# inside (name of person) is a picture of know faces of this person
CWD_DIR = os.getcwd()
KNOWN_FACE_DIR = os.path.join(CWD_DIR,'known_faces')

# Machice learning model
MODEL = 'cnn'

# Create empty list to contain know face and name 
print('Encoding faces...')

encoding_data = {}

ENCODE_FILENAME = 'encode_faces.pkl'

## walk in know face directory
for person_name in os.listdir(KNOWN_FACE_DIR):
    # person_name is name of a known face person folder. also the name of the that person (use to label later)
    pic_person_folder = os.path.join(KNOWN_FACE_DIR,person_name)  # Full path of person folder
    
    known_faces = []

    # Iterate all picture for eack person 
    for pic_person_filename in os.listdir(pic_person_folder):
        pic_person_filepath = os.path.join(pic_person_folder,pic_person_filename) # Full path of person picture
        #Load an image 
        person_image = face_recognition.load_image_file(pic_person_filepath)

        location_face = face_recognition.face_locations(person_image,model=MODEL)
        print(location_face)
        show = person_image[location_face[0][0]:location_face[0][2],location_face[0][3]:location_face[0][1]]
        show = cv2.cvtColor(show,cv2.COLOR_RGB2BGR)
        cv2.imshow(person_name,show)
        cv2.waitKey(1)

        # Get 128-dimension face encoding
        # face_recognition.face_encodings return every face in image
        # Always returns a list of found faces, For this purpose we take first face only 
        # Assuming one face per image as you can't be twice on one image
        # Need to make sure that train dataset have one face in image
        encoding_face = face_recognition.face_encodings(person_image,location_face)[0] # index 0 is the first face found

        # Append encoding and name
        # will be match index together
        known_faces.append(encoding_face)
    
    # Store data in dictionary base in form of person name : encoding face
    encoding_data[person_name] = known_faces

print('DONE!')

for person_name , num_face in encoding_data.items():
    print('Encoding {} faces of person name {}'.format(len(num_face),person_name))



with open(ENCODE_FILENAME,'wb') as pickle_file:
    pickle.dump(encoding_data,pickle_file)

