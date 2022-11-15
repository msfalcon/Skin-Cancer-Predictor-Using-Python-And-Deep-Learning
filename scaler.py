
#Image Scaling Function


def img_scaler(i):
    import numpy as np
    test1=np.asarray(Image.open(i).resize((32,32))) #OPening the image, resizing it and forming a numpy array of the input image
    s_img=test1/225
    image_resize = np.expand_dims(s_img, axis=0)
    return s_img