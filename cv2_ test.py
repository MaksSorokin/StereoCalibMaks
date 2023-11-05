import cv2


def initPath():
    imgArr=['calib_row\\img0m5.bmp', 'calib_row\\img00.bmp', 'calib_row\\img05.bmp']
    return imgArr


def loading_displaying_saving():
    img = cv2.imread(initPath()[0], cv2.IMREAD_GRAYSCALE)
    cv2.imshow('calib', img)
    print("Высота:"+str(img.shape[0]))
    print("Ширина:" + str(img.shape[1]))
    cv2.waitKey(0)
    cv2.imwrite('calib.bmp', img)


loading_displaying_saving()
