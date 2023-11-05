import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
 
def calibrate_camera(images_folder, rows, columns, world_scaling, camNumber):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
    binImages = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binImage = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        binImage = cv.blur(binImage, (3,3))
        binImages.append(binImage)
        #cv.imshow("image", binImage)
        #cv.waitKey(0)
 
    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    flagWriteCorn = 0
    for frame in binImages:
        #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = frame
        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            #cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            #cv.imshow('img', frame)
            k = cv.waitKey(0)
 
            objpoints.append(objp)
            imgpoints.append(corners)
            print("corners:")
            print(np.shape(corners))
            #with open(str(flagWriteCorn) + str(camNumber) + "corners.txt", "w") as fp:
                #for item in corners:
                    ## write each item on a new line
                    #fp.write("%s\n" % item[0])
            np.save(str(flagWriteCorn) + str(camNumber) + "corners", corners)
        flagWriteCorn +=1
    
    
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
 
#mtx1, dist1 = calibrate_camera(images_folder = 'D2/*')
#mtx2, dist2 = calibrate_camera(images_folder = 'J2/*')


def stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_folder1, frames_folder2, rows, columns, world_scaling):
    #read the synched frames
    images_names1 = glob.glob(frames_folder1)
    images_names2 = glob.glob(frames_folder2)
 
    c1_images = []
    c2_images = []
    for im1, im2 in zip(images_names1, images_names2):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)
 
        _im = cv.imread(im2, 1)
        c2_images.append(_im)
    
    binImages1 = []
    binImages2 = []
    
    for image in c1_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binImage1 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        binImage1 = cv.blur(binImage1, (3,3))
        binImages1.append(binImage1)
        #cv.imshow("image", binImage1)
        #cv.waitKey(0)
    for image in c2_images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        binImage2 = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        binImage2 = cv.blur(binImage2, (3,3))
        binImages2.append(binImage2)
        #cv.imshow("image", binImage2)
        #cv.waitKey(0)
 
    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
 
 
    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp
 
    #frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]
 
    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []
 
    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space
 
    for frame1, frame2 in zip(binImages1, binImages2):
        #gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        #gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)\
        gray1 = frame1
        gray2 = frame2
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
 
        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv.drawChessboardCorners(frame1, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame1)
 
            cv.drawChessboardCorners(frame2, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame2)
            k = cv.waitKey(0)
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
 
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T
 
#R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, 'synched/*')


def triangulate (mtx1, mtx2, cam1Points, cam2Points, R, T):
 
    #uvs1 = [[458, 86], [451, 164], [287, 181],
            #[196, 383], [297, 444], [564, 194],
            #[562, 375], [596, 520], [329, 620],
            #[488, 622], [432, 52], [489, 56]]
 
    #uvs2 = [[540, 311], [603, 359], [542, 378],
            #[525, 507], [485, 542], [691, 352],
            #[752, 488], [711, 605], [549, 651],
            #[651, 663], [526, 293], [542, 290]]
    
    uvs1 = np.load(cam1Points)
    uvs2 = np.load(cam2Points)
    uvs1List = [x[0] for x in uvs1]
    uvs2List = [x[0] for x in uvs2]
    uvs1 = np.array(uvs1List)
    uvs2 = np.array(uvs2List)


 
    frame1 = cv.imread('imgCheckmate\\cam1b\\1dz20.bmp')
    frame2 = cv.imread('imgCheckmate\\cam2b\\2dz20.bmp')
 
    plt.imshow(frame1[:,:,[2,1,0]])
    plt.scatter(uvs1[:,0], uvs1[:,1])
    plt.show() #this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this.
 
    plt.imshow(frame2[:,:,[2,1,0]])
    plt.scatter(uvs2[:,0], uvs2[:,1])
    plt.show()#this call will cause a crash if you use cv.imshow() above. Comment out cv.imshow() to see this
 
    #RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = mtx1 @ RT1 #projection matrix for C1
 
    #RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis = -1)
    P2 = mtx2 @ RT2 #projection matrix for C2
 
    def DLT(P1, P2, point1, point2):
 
        A = [point1[1]*P1[2,:] - P1[1,:],
             P1[0,:] - point1[0]*P1[2,:],
             point2[1]*P2[2,:] - P2[1,:],
             P2[0,:] - point2[0]*P2[2,:]
            ]
        A = np.array(A).reshape((4,4))
        #print('A: ')
        #print(A)
 
        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices = False)
 
        print('Triangulated point: ')
        print(Vh[3,0:3]/Vh[3,3])
        return Vh[3,0:3]/Vh[3,3]
 
    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
 
    from mpl_toolkits.mplot3d import Axes3D
 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xMinMaxList = [p3ds[x][0] for x in range(np.shape(p3ds)[0])]
    yMinMaxList = [p3ds[x][1] for x in range(np.shape(p3ds)[0])]
    zMinMaxList = [p3ds[x][2] for x in range(np.shape(p3ds)[0])]
    ax.set_xlim3d(min(xMinMaxList), max(xMinMaxList))
    ax.set_ylim3d(min(yMinMaxList), max(yMinMaxList))
    ax.set_zlim3d(min(zMinMaxList), max(zMinMaxList))
 
    #connections = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,8], [1,9], [2,8], [5,9], [8,9], [0, 10], [0, 11]]
    for point in p3ds:
        #ax.plot(xs = point[0], ys = point[2], zs = point[3], c = 'red')
        ax.scatter(point[0], point[1], point[2])
    ax.set_title('This figure can be rotated.')
    #uncomment to see the triangulated pose. This may cause a crash if youre also using cv.imshow() above.
    plt.show()
