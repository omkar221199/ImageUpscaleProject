
####### CORE IMPORTS ###############
import cv2

from cv2 import dnn_superres
import streamlit as st
from PIL import Image,ImageEnhance
import time
import numpy as np
import os
import random
import string
from datetime import datetime
def upScaleEDSR(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()
    path = "ImageUpscaleProject/weights/EDSR_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("edsr", 3)
    result = sr.upsample(image)
    return result
    #cv2.imwrite('SuperResTest/UpscaledIm/'+ saveName, result)


def upScaleFSRCNN(image):
    # Create an SR object
    sr = dnn_superres.DnnSuperResImpl_create()

    path = "ImageUpscaleProject/weights/FSRCNN_x3.pb"
    sr.readModel(path)
    # Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("fsrcnn", 3)
    result = sr.upsample(image)
    return result
    #cv2.imwrite("SuperResTest/UpscaledIm/"+saveName, result)
    
    
def DownGrade(imagepath,factor = 0.5):
    img = cv2.imread(imagepath)
    img = cv2.resize(img, (0, 0), fx = factor, fy = factor) 
    #img = cv2.resize(img, (size, size), interpolation = cv2.INTER_LINEAR)
    #img = cv2.resize(img,(size,size))
    cv2.imwrite('SuperResTest/downscaled.png',img)


    
def BicubicUpscaling(img,factor = 2):
    
    img_resized = cv2.resize(img, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return img_resized
    #cv2.imwrite('SuperResTest/bicubicUp/'+saveName,img)
    
def denoise(img):

    # denoising of image saving it into dst image 
    dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    return dst
    
def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def save(typ,img):
    now = datetime.now()
    time_stamp = now.strftime("%m_%d_%H_%M_%S") 
    fn ='ImageUpscaleProject/Saved_Images/'+typ+time_stamp+'.png'
    cv2.imwrite(fn,img)
    
    
def main():
    st.title("Image Super Sampling and Denoiser")
    st.subheader("App for AI based Image super resolution with EDSR and FSRCNN models for image upscaling and Denoising.")
    
    #Uploading Main Image
    img_file = st.file_uploader("Browse your Image or Drag and Drop",type = ['jpg','jpeg','png'])
    if img_file is None:
        st.info("Please Select an Image")
    else:
        
        #Convert Selected image to opencv format
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        main_img_cv = cv2.imdecode(file_bytes, 1)
        st.image(main_img_cv,channels = 'BGR',use_column_width = True)

        # Sidebar activites defined
        activities = ['Super Sampling','Denoise','Resize','Filter','Crop','About']
        sidebar_choice = st.sidebar.selectbox('Select a feature',activities)
        #Sidebar activites
        if sidebar_choice == 'Super Sampling':
            st.subheader("Image Super resolution")
            upscale_type = st.selectbox("Select Upsampling Method",['EDSR','FSRCNN','Bicubic'])
            if upscale_type == 'EDSR':
                st.info('The Enhanced deep residual super sampling method is based on a larger model and will take anywhere from 2 - 15 min to complete upsampling.')
                if st.button("Apply and Save"):
                    if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
                        st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
                    else:
                        with st.spinner("Upscaling. Please Wait. This may take long."):
                            upscaled_image_cv = upScaleEDSR(main_img_cv)
                        
                        st.image(upscaled_image_cv, channels="BGR")
                        st.success("Image has been Upscaled")
                        save('EDSR_',upscaled_image_cv)
                        st.success("File has been Saved")
                        st.balloons()

            elif upscale_type == 'FSRCNN':
                st.info('FSRCNN is small fast model of quickly upscaling images with AI. This model does not produce state of the art accuracy however')
                
                if st.button("Apply and Save"):
                    if main_img_cv.shape[0] > 1080 or main_img_cv.shape[1] > 1080:
                        st.error('Image to Large for upsampling. This image is already above 1080 pixels wide')
                    else:
                        with st.spinner("Upscaling... Hold on Tight this may take a cuple of seconds"):
                            upscaled_image_cv = upScaleFSRCNN(main_img_cv)
                        st.image(upscaled_image_cv, channels="BGR",format = 'png')
                        st.success("Image has been Upscaled")
                        save('FSRCNN_',upscaled_image_cv)
                        st.success("File has been Saved")
                    
        
        
        elif sidebar_choice == 'Denoise':
            st.subheader("Image Denoiser")
            st.info('The Image denoiser will remove noise from Images. Implementation via OpenCV')
            if st.button("Apply and Save"):
                with st.spinner("Denoising.. Please Hold On to your seat belts"):
                    denoise_image_cv = denoise(main_img_cv)
                    st.image(denoise_image_cv,channels="BGR",use_column_width = True)
                #ranstr = randomString(6)
                save('Denoise_',denoise_image_cv)
                #cv2.imwrite('ImageUpscaleProject/Saved_Images/Denoised_'+ranstr+'.png',denoise_image_cv)
                st.success("Image was Denoised and saved")
                st.balloons()


    
    
if __name__ == '__main__':
    main()
    


