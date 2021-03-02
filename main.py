import cv2 #librari per leximin e fotove ose videove
import imutils
import time
import datetime
import pafy
import sys
import numpy as np
zgjedhja = input("Zgjedhni metoden e detektimit te fytyrave: \na) Me kamere \nb) Me link te youtubes\nZgjedhni ketu: ")
if (zgjedhja == "1"):
    # per kamere
    cap = cv2.VideoCapture(0)
elif(zgjedhja == "2"):
    linku = input("Shkruani linkun e youtubes: ")
    #merr linkun e youtubes
    #url = 'https://www.youtube.com/watch?v=c07IsbSNqfI&feature=youtu.be'
    vPafy = pafy.new(str(linku))
    play = vPafy.getbest(preftype="mp4")
    #per video te youtubes
    cap = cv2.VideoCapture(play.url)
else:
    print("Gabim ne input")
    sys.exit()


'''
cap.set(3,1280)
cap.set(4,1024)
cap.set(15, 0.1)'''
cap.set(3, 480)  # Gjeresia
cap.set(4, 640)  # Gjatesia
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Mashkull', 'Femer']
datetime = datetime.date.today().strftime('%m/%d/%y')
t = time.localtime()
#current_time = time.strftime("%H:%M:%S", t)
current_time = time.strftime("%H:%M", t)
datetime += str(" "+current_time)

#modele te para-trajnuara te CNN te cilat do te bejne detektimin.
def initialize_caffe_models():
    # klasifiko gjinine
    age_net = cv2.dnn.readNetFromCaffe(
        'deploy_age.prototxt',
        'age_net.caffemodel')
    #klasifiko moshen
    gender_net = cv2.dnn.readNetFromCaffe(
        'deploy_gender.prototxt',
        'gender_net.caffemodel')

    return (age_net, gender_net)

def read_from_camera(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:

        ret, image = cap.read()

        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        #Paraqitja e dates
        cv2.putText(image, str(datetime), (10, 50), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        if (len(faces) > 0):
            print("U gjeten {} fytyra".format(str(len(faces))))
        cv2.putText(image, "U gjeten {} fytyra".format(str(len(faces))), (10, 80), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Merr fytyren
            face_img = image[y:y + h, h:h + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Qello Gjinine
            gender_net.setInput(blob)
            gender_preds = gender_net.forward() #kthen mundesite dhe koordinatat
            gender = gender_list[gender_preds[0].argmax()]
            print("Gjinia : " + gender)

            # Qello moshen
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print("Rangu i moshes: " + age)

            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        '''cv2.putText(image, "Shtyp 'q' per te mbyllur programin".format(str(len(faces))), (10, 110), font, 0.5,
                    (255, 255, 255), 2,
                    cv2.LINE_AA)'''
        cv2.imshow('Detektimi i fytyres', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    age_net, gender_net = initialize_caffe_models()

    read_from_camera(age_net, gender_net)
