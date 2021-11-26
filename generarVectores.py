import face_recognition
import os
file = open("vectores.csv", "w")
folder_path = "fotos_bd\lfw"
folders = os.listdir(folder_path)

for folder in folders:
    images = os.listdir(folder_path + '/' + folder)
    for image in images:
        path = folder_path + '/' + folder + '/' + image
        picture = face_recognition.load_image_file(path)
        vector = face_recognition.face_encodings(picture)
        try:
            line = ""
            for i in range(128-1):
                line+= str(vector[0][i]) + ","
            line+= str(vector[0][127]) + "\n"
            file.write(line)
        except:
            print("Fail")
file.close()

