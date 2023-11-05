import cv2


def initPath():
    imgArr=['calib_row\\img0m5.bmp', 'calib_row\\img00.bmp', 'calib_row\\img05.bmp']
    return imgArr


def loading_displaying_saving():
    img = cv2.imread(initPath()[0], cv2.IMREAD_GRAYSCALE)
    cv2.imshow('calib', img)
    cv2.waitKey(0)
    cv2.imwrite('calib.bmp', img)


def resizing():
    img = initPath()[0]
    res_img_nearest = cv2.resize(img, (int(w / 1.4), int(h / 1.4)), 
                                cv2.INTER_NEAREST)
    res_img_linear = cv2.resize(img, (int(w / 1.4), int(h / 1.4)), 
                                cv2.INTER_LINEAR)
    cv2.imshow('1', res_img_nearest)
    cv2.imshow('2', res_img_linear)
    cv2.waitKey(0)
    
calibList = [ret, mtx, dist, rvecs, tvecs]
calibTxtList = [str(x) for x in calibList]

with open("calib0.txt", "w") as output:
    for row in calibTxtList:
        output.write(str(row) + '\n')

resizing()
