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
        #binImage = cv.blur(binImage, (3,3))
        #binImage = gray
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
            conv_size = (23, 23)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
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
 
            #cv.drawChessboardCorners(frame1, (rows,columns), corners1, c_ret1)
            #cv.imshow('img', frame1)
 
            #cv.drawChessboardCorners(frame2, (rows,columns), corners2, c_ret2)
            #cv.imshow('img2', frame2)
            #k = cv.waitKey(0)
 
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
    
    #uvs1 = np.load(cam1Points)
    #uvs2 = np.load(cam2Points)
    #uvs1List = [x[0] for x in uvs1]
    #uvs2List = [x[0] for x in uvs2]
    #uvs1 = np.array(uvs1List)
    #uvs2 = np.array(uvs2List)

    #for file.txt:
 
    # Open the file in read mode
    with open(cam1Points, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Create an empty array with the desired shape
    uvs1 = np.zeros((2, len(lines)))

    # Iterate through each line and extract the two numbers
    for i, line in enumerate(lines):
        # Split the line into two numbers
        numbers = line.strip().split(' ')
    
        # Store the two numbers in the array
        uvs1[0, i] = float(numbers[0])
        uvs1[1, i] = float(numbers[1])

    with open(cam2Points, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

        # Create an empty array with the desired shape
    uvs2 = np.zeros((2, len(lines)))

        # Iterate through each line and extract the two numbers
    for i, line in enumerate(lines):
        # Split the line into two numbers
        numbers = line.strip().split(' ')
        
        # Store the two numbers in the array
        uvs2[0, i] = float(numbers[0])
        uvs2[1, i] = float(numbers[1])

    uvs10 = []
    uvs20 = []

    for i in range(uvs1.shape[1]):
        uvs10.append([uvs1[0][i], uvs1[1][i]])
    for i in range(uvs1.shape[1]):
        uvs20.append([uvs2[0][i], uvs2[1][i]])
    
    uvs1 = uvs10
    uvs2 = uvs20
    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)
    # Print the array
    print(uvs10)
    
    uvs1 = np.array(uvs1)
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


def flannBasedMatcher(image1, image2):

    img1 = cv.imread(image1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(image2, cv.IMREAD_GRAYSCALE)
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    img4=cv.drawKeypoints(img1, kp1, img1)
    img5=cv.drawKeypoints(img2, kp2, img2)
    # FLANN parameters
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #plt.imshow(img3)
    plt.imshow(img4)
    plt.imshow(img5)
    plt.show()


def findCircle(imageName):
    # Load image 
    image = cv.imread(imageName, 0) 
  
    # Set our filtering parameters 
    # Initialize parameter setting using cv.SimpleBlobDetector 
    params = cv.SimpleBlobDetector_Params() 
  
    # Set Area filtering parameters 
    params.filterByArea = False
    params.minArea = 100
  
    # Set Circularity filtering parameters 
    params.filterByCircularity = False 
    params.minCircularity = 0.1
  
    # Set Convexity filtering parameters 
    params.filterByConvexity = False
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters 
    params.filterByInertia = False
    params.minInertiaRatio = 0.01
    
    # Create a detector with the parameters 
    detector = cv.SimpleBlobDetector_create(params) 
        
    # Detect blobs 
    keypoints = detector.detect(image) 
    
    # Draw blobs on our image as red circles 
    blank = np.zeros((1, 1))  
    blobs = cv.drawKeypoints(image, keypoints, blank, (0, 0, 255), 
                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 
    
    number_of_blobs = len(keypoints) 
    text = "Number of Circular Blobs: " + str(len(keypoints)) 
    cv.putText(blobs, text, (20, 550), 
                cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2) 
    
    # Show blobs 
    cv.imshow("Filtering Circular Blobs Only", blobs) 
    cv.waitKey(0) 
    cv.destroyAllWindows() 


def featureMatch(imageName1, imageName2):
    # read the images
    img1 = cv.imread(imageName1)  
    img2 = cv.imread(imageName2)

    # convert images to grayscale
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # create SIFT object
    sift = cv.xfeatures2d.SIFT_create()
    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2,None)
    # create feature matcher
    bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)
    # match descriptors of both images
    matches = bf.match(descriptors_1,descriptors_2)
    # sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    # draw first 50 matches
    matched_img = cv.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)
    # show the image
    cv.imshow('image', matched_img)
    # save the image
    cv.imwrite("matched_images.jpg", matched_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def calibrate_cameraCircle(images_folder, rows, columns, world_scaling, camNumber):
    images_names = glob.glob(images_folder)
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)
        images.append(im)
    binImages = []
    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #binImage = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]
        #binImage = cv.blur(binImage, (3,3))
        binImage = gray
        binImages.append(binImage)
        cv.imshow("image", binImage)
        cv.waitKey(0)
 
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
        params = cv.SimpleBlobDetector_Params()
        params.maxArea = 10000
        detector = cv.SimpleBlobDetector_create(params)
        ret, corners = cv.findCirclesGrid(gray, (rows, columns),None, None, detector)
 
        if ret == True:
 
            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)
 
            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.imshow('img', frame)
            k = cv.waitKey(0)
 
            objpoints.append(objp)
            imgpoints.append(corners)
            print("corners:")
            print(np.shape(corners))
            #with open(str(flagWriteCorn) + str(camNumber) + "corners.txt", "w") as fp:
                #for item in corners:
                    ## write each item on a new line
                    #fp.write("%s\n" % item[0])
            np.save(str(flagWriteCorn) + str(camNumber) + "cornersCircle", corners)
        flagWriteCorn +=1
    
    
 
 
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', mtx)
    print('distortion coeffs:', dist)
    print('Rs:\n', rvecs)
    print('Ts:\n', tvecs)
 
    return mtx, dist
