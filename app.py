from distutils.log import debug
from flask import Flask

#Importing linraries
import numpy as np
from PIL import Image

#Loading Skin cancer model
from tensorflow.keras.models import load_model
cancer_model=load_model("skin_cancer_model.h5")


#image input
input_img1="C:\Users\D-19-CS-11\FYP 1\Test_images\img2.jpg"

#Prediction Model
def prediction_model(input_img,cancer_model):

    #Resizing and scaling function
    def img_scaler(input_img):
        test1=np.asarray(Image.open(input_im).resize((32,32))) #OPening the image, resizing it and forming a numpy array of the input image
        s_img=test1/225
        image_resize = np.expand_dims(s_img, axis=0)
        return image_resize
    
    class_name=np.array(['Actinic keratoses (akiec)', 'Basal cell carcinoma (bcc)', 'Benign Keratosis-like lesions (bkl)', 'Melanoma (mel)', 'Melanocytic nevi (nv)', 'Vascular lesions (vas)'])
    
    neck_samp=image_resize
    v1=img_scaler(neck_samp)
    
    out=cancer_model.predict_classes(v1)[0]    #predict classes will predict the classes. Otherwise it will show some normalized data
                                               #returning first index
        
    return class_name[out]

app=Flask(__name__)

@app.route("/")
def home():
    return "<h1> Hello World</h1>"

cancer_model=load_model("skin_cancer_model.h5")


@app.route("/api/cancer",methods=["POST"])
def cancer_predictor():
    input_image=input_img1
    pred=prediction_model(input_image,cancer_model,)

    return pred

if __name__=="__main__":
    app.run()