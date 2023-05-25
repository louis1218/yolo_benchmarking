import cv2
import numpy as np
import sys
sys.path.insert(0,'..')
from Naoqi_camera import NaoqiCamera


if __name__ == "__main__":
    naoqi_camera = NaoqiCamera(ip="tcp://192.168.50.44:9559")
    al_motion = naoqi_camera.session.service("ALMotion")
    counter = 0
    while cv2.waitKey(1) != 27:
        al_motion.setAngles("HeadPitch", 0.0,0.1)
        al_motion.setAngles("Headyaw", 0.0,0.1)
        img = naoqi_camera.get_image()
        cv2.imshow('image',img)
        cv2.imwrite('images/'+counter+'.jpeg',img)
        counter +=1

        al_motion.setAngles("HeadPitch", 0.2,0.1)
        al_motion.setAngles("Headyaw", 0.2,0.1)
        img = naoqi_camera.get_image()
        cv2.imshow('image',img)
        cv2.imwrite('images/'+counter+'.jpeg',img)
        counter +=1

        al_motion.setAngles("HeadPitch", -0.2,0.1)
        al_motion.setAngles("Headyaw", 0.2,0.1)
        img = naoqi_camera.get_image()
        cv2.imshow('image',img)
        cv2.imwrite('images/'+counter+'.jpeg',img)
        counter +=1

        al_motion.setAngles("HeadPitch", -0.2,0.1)
        al_motion.setAngles("Headyaw", -0.2,0.1)
        img = naoqi_camera.get_image()
        cv2.imshow('image',img)
        cv2.imwrite('images/'+counter+'.jpeg',img)
        counter +=1

        al_motion.setAngles("HeadPitch", 0.2,0.1)
        al_motion.setAngles("Headyaw", -0.2,0.1)
        img = naoqi_camera.get_image()
        cv2.imshow('image',img)
        cv2.imwrite('images/'+counter+'.jpeg',img)
        counter +=1

    cv2.destroyAllWindows()
    naoqi_camera.cleanup()
