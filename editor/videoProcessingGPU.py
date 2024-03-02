import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
import os 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import time
from math import atan2, degrees, cos, sin, radians
import traceback 
import colour
from colormath.color_diff import delta_e_cie2000
from skimage.metrics import structural_similarity
import imutils
from skimage.color import rgb2lab, deltaE_ciede2000
import json
from config import *
import sys
# ### ORB
### Source: https://github.com/armaanpriyadarshan/Rotation-Tracking-OpenCV/blob/main/main.py
# template_img_bw = cv2.imread(os.path.join(BEE_TEMPLATES_DIR, 'beeVideo2.jpg'), cv2.COLOR_BGR2GRAY) # queryImage
# train_img_bw = cv2.imread(os.path.join(PROJECT_DIR, 'frame113.jpg'),cv2.COLOR_BGR2GRAY) # trainImage

class VideoProcessingGPU():
    def __init__(self, show=False, program_dir=None):
        self.program_dir = program_dir
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        if load_prog['typeId'] == 'A':
            self.full_template = cv2.imread(os.path.join(program_dir, "templates/full.jpg"))
            self.crayon_template = cv2.imread(os.path.join(program_dir, "templates/crayon.jpg"))
        elif load_prog['typeId'] == 'UPG':
            self.full_template = cv2.imread(os.path.join(program_dir, "templates/10ports.jpg"))
            self.gpu_full_template = cv2.cuda_GpuMat()
            self.gpu_full_template.upload(self.full_template)

        self.show = show

    def orb(self, template_img, original_img):
        try:
            h, w, layers = template_img.shape
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) 
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 
            
            template_gray = cv2.medianBlur(template_gray, 5)
            img_gray = cv2.medianBlur(img_gray, 5)

            # Initialize the ORB detector algorithm 
            orb = cv2.ORB_create() 
            
            # Now detect the keypoints and compute 
            # the descriptors for the query image 
            # and train image 
            queryKeypoints, queryDescriptors = orb.detectAndCompute(template_gray, None) 
            trainKeypoints, trainDescriptors = orb.detectAndCompute(img_gray, None) 
            
            # Initialize the Matcher for matching 
            # the keypoints and then match the 
            # keypoints 
            # matcher = cv2.BFMatcher() 
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            matches = matcher.match(queryDescriptors, trainDescriptors) 

            matches = sorted(matches, key=lambda x: x.distance)

            template_pts = np.float32([queryKeypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            img_pts = np.float32([trainKeypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            rect = cv2.minAreaRect(transformed_corners)

            return rect
        
        except:
            return None

    def akaze_gpu(self, template_img, original_img):
        try:
            h, w, layers = template_img.shape
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) 
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 
            
            template_gray = cv2.medianBlur(template_gray, 5)
            img_gray = cv2.medianBlur(img_gray, 5)

            akaze = cv2.cuda.AKAZE.create()

            # Find the keypoints and descriptors with AKAZE
            keypoints1, descriptors1 = akaze.detectAndCompute(template_gray, None)
            keypoints2, descriptors2 = akaze.detectAndCompute(img_gray, None)

            # Convert AKAZE descriptors to CV_32F
            descriptors1 = descriptors1.astype('float32')
            descriptors2 = descriptors2.astype('float32')

            # Upload keypoints and descriptors to the GPU
            keypoints1_gpu = cv2.cuda_GpuMat()
            keypoints1_gpu.upload(cv2.KeyPoint_convert(keypoints1))
            descriptors1_gpu = cv2.cuda_GpuMat()
            descriptors1_gpu.upload(descriptors1)
            keypoints2_gpu = cv2.cuda_GpuMat()
            keypoints2_gpu.upload(cv2.KeyPoint_convert(keypoints2))
            descriptors2_gpu = cv2.cuda_GpuMat()
            descriptors2_gpu.upload(descriptors2)

            # Initialize CUDA Brute-Force Matcher
            bf_matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)

            # Match descriptors on the GPU
            matches_gpu = bf_matcher.knnMatch(descriptors1_gpu, descriptors2_gpu, k=2)

            # Download matches back to the CPU
            matches = [m[0] for m in matches_gpu]
            matches = sorted(matches, key=lambda x: x.distance)

            template_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            img_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            rect = cv2.minAreaRect(transformed_corners)

            return rect
        
        except Exception as e:
            print("Exception @ akaze @ line", sys.exc_info()[2].tb_lineno, sys.exc_info())
            return None

    def orb_gpu(self, template_img, original_img):
        try:

            MAX_FEATURES = 9000              
            GOOD_MATCH_PERCENT = 0.15
            hessian_threshold = 800
            h, w, layers = template_img.shape
            

            # Upload images to GPU
            gpu_image1 = cv2.cuda_GpuMat()
            gpu_image2 = cv2.cuda_GpuMat()
            gpu_image1.upload(template_img)
            gpu_image2.upload(original_img)

            gpu_image1 = cv2.cuda.cvtColor(gpu_image1, cv2.COLOR_RGB2GRAY)
            gpu_image2 = cv2.cuda.cvtColor(gpu_image2, cv2.COLOR_RGB2GRAY)

            # Create SURF_CUDA objects with specified hessianThreshold values
            hessian_threshold = 800  # Adjust this value based on your needs
            surf = cv2.cuda.SURF_CUDA.create(_hessianThreshold=hessian_threshold)

            # Detect and compute keypoints and descriptors on GPU
            keypoints_gpu1, descriptors_gpu1 = surf.detectWithDescriptors(gpu_image1, None)
            keypoints_gpu2, descriptors_gpu2 = surf.detectWithDescriptors(gpu_image2, None)

            # Download keypoints and descriptors from GPU
            kp1 = keypoints_gpu1.download()
            descriptors1 = descriptors_gpu1.download()
            kp2 = keypoints_gpu2.download()
            descriptors2 = descriptors_gpu2.download()

            # Use the BFMatcher to find the best matches
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            # Download matches back to the CPU
            matches = [m[0] for m in matches]
            matches = sorted(matches, key=lambda x: x.distance)
            # Check if there are enough good matches to compute homography
            if len(matches) >= 4:
                # # Extract matched keypoints
                # src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                # dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Extract matched keypoints
                src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


                # Use findHomography to compute the homography matrix
                homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


                corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(corners, M)

                rect = cv2.minAreaRect(transformed_corners)
            

            return rect
        
        except Exception as e:
            print("Exception @ akaze @ line", sys.exc_info()[2].tb_lineno, sys.exc_info())
            return None

    def crop_rotate(self, img, rect):
        center, size, angle = rect[0], rect[1], rect[2]
        center, size = tuple(map(int, center)), tuple(map(int, size))
        # print(center, size, angle)
        rotated = False
        height, width = img.shape[0], img.shape[1]

        if angle > 45 and angle < 90:
            angle -= 90
            rotated = True
        # calculate the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # rotate the original image
        img_rot = cv2.warpAffine(img, M, (width, height))
        # cv2.imwrite(os.path.join(GS_DIR, "img_rot.jpg"), img_rot)

        W = rect[1][0]
        H = rect[1][1]

        croppedW = W if not rotated else H 
        croppedH = H if not rotated else W
        y1 = np.max([0, int(center[1]-croppedH/2)])
        y2 = np.min([img_rot.shape[0], int(center[1]+croppedH/2)])

        x1 = np.max([0, int(center[0]-croppedW/2)])
        x2 = np.min([img_rot.shape[1], int(center[0]+croppedW/2)])
        img_rot_ROI = img_rot[y1:y2, x1:x2]
        # cv2.imwrite(os.path.join(GS_DIR, "img_rot_ROI.jpg"), img_rot_ROI)

        return img_rot_ROI

    def sift(self, template_img, original_img):
        try:
            template_img = cv2.medianBlur(template_img, 3)
            original_img = cv2.medianBlur(original_img, 3)

            h, w, layers = template_img.shape

            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) 
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 

            gpu_image = cv2.cuda_GpuMat()
            gpu_image.upload(img_gray)
            gpu_template = cv2.cuda_GpuMat()
            gpu_template.upload(template_gray)
            
            # Initiate SIFT detector
            sift = cv2.cuda.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1_gpu, des1_gpu = sift.detectAndCompute(gpu_template,None)
            kp2_gpu, des2_gpu = sift.detectAndCompute(gpu_image,None)
            kp1 = kp1_gpu.download()
            kp2 = kp2_gpu.download()

            # Initialize the Matcher for matching 
            # the keypoints and then match the 
            # keypoints 
            matcher_gpu = cv2.cuda_BFMatcher.create() 
            matches_gpu = matcher.match(des1_gpu, des2_gpu) 
            matches = matches_gpu.download()
            matches = sorted(matches, key=lambda x: x.distance)

            template_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            img_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            rect = cv2.minAreaRect(transformed_corners)

            return rect
        except Exception as e:
            print("Exception @ sift @ line", sys.exc_info()[2].tb_lineno, sys.exc_info())
            return None

    def templateMatching(self, template_img, original_img, threshold=0.8):
        template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) 
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 

        detected = False

        assert template_gray is not None, "file could not be read, check with os.path.exists()"
        w, h = template_gray.shape[::-1]
        res = cv2.matchTemplate(img_gray,template_gray,cv2.TM_CCOEFF_NORMED)

        (yCoords, xCoords) = np.where( res >= threshold)

        if np.any((yCoords, xCoords)):
            rects = []
            # loop over the starting (x, y)-coordinates again
            for (x, y) in zip(xCoords, yCoords):
                # update our list of rectangles
                rects.append((x, y, x + w, y + h))
            # apply non-maxima suppression to the rectangles
            pick = non_max_suppression(np.array(rects))
            # print("[INFO] {} matched locations *after* NMS".format(len(pick)))
            # loop over the final bounding boxes
            for (startX, startY, endX, endY) in pick:
                # draw the bounding box on the image
                cv2.rectangle(original_img, (startX, startY), (endX, endY), (255, 0, 0), 3)

            detected = 1
        return original_img, detected

    def create_point_from_origin(self, center, standard, dist, angle):
        alpha = degrees(atan2(standard[1]-center[1], standard[0]-center[0]))
        beta = angle + alpha
        output_point = (center[0] + int(dist*cos(radians(beta))), center[1] + int(dist*sin(radians(beta))))
        return output_point

    def crop_from_center(self, image, center, w, h):
        return image[int(center[1]-w/2):int(center[1]+w/2),int(center[0]-h/2):int(center[0]+h/2)]

    def compare_2_images_hsv(self, image1, image2):
        (h1, w1) = image1.shape[:2]
        (h2, w2) = image2.shape[:2]
        h, w = None, None
        # check the height
        h = np.min([h1, h2])
        #check the width
        w = np.min([w1, w2])

        image1 = cv2.resize(image1, (w,h))
        image2 = cv2.resize(image2, (w,h))

        img1_hsv = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
        num = 6
        if ROTATE:
            step        = int(w/num)
            step_dis    = int(step/2)
        else:
            step        = int(h/num)
            step_dis    = int(step/2)
        dis = 0
        metric_val_list = []
        for i in range(2*num-1):
            if ROTATE:
                image1_splitted = img1_hsv[0:h, dis:dis+step]
                # cv2.imwrite(os.path.join(IMAGES_DIR, "image1_splitted_{}.jpg".format(i)), image1_splitted)
                
                image2_splitted = img2_hsv[0:h, dis:dis+step]
                # cv2.imwrite(os.path.join(IMAGES_DIR, "image2_splitted_{}.jpg".format(i)), image2_splitted)
            else:
                image1_splitted = img1_hsv[dis:dis+step, 0:w]
                # cv2.imwrite(os.path.join(IMAGES_DIR, "image1_splitted_{}.jpg".format(i)), image1_splitted)
                
                image2_splitted = img2_hsv[dis:dis+step, 0:w]
                # cv2.imwrite(os.path.join(IMAGES_DIR, "image2_splitted_{}.jpg".format(i)), image2_splitted)
            
            # Calculate the histogram and normalize it
            hist_img1 = cv2.calcHist([image1_splitted], [0,1], None, [180,256], [0,180,0,256])
            cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_img2 = cv2.calcHist([image2_splitted], [0,1], None, [180,256], [0,180,0,256])
            cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            # find the metric value
            metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
            # print(i, metric_val)
            metric_val_list.append(metric_val)
            dis += step_dis

        return metric_val_list

    def process(self, image):
        FULL_DETECTED = False
        BEE_DETECTED = False
        CAT_DETECTED = False
        CRAYON_DETECTED = False
        percent = 0
        jugdement_value = 0

        if ROTATE:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        CORRECT = False
        # image = image[CONVEY_AREA_Y_RANGE[0]:COports_templateNVEY_AREA_Y_RANGE[1], CONVEY_AREA_X_RANGE[0]:CONVEY_AREA_X_RANGE[1]]
        try:
            image[int(image.shape[0]-self.crayon_template.shape[0]):image.shape[0],
                    int(image.shape[1]/2-self.crayon_template.shape[1]/2):int(image.shape[1]/2+self.crayon_template.shape[1]/2)] = self.crayon_template
            ### ORB
            full_rect = self.orb(self.full_template, image)
            if full_rect == None: 
                CORRECT = "full_rect"
                print("NO BOOK -> Exception")
            else:
                full_rect_area = full_rect[1][0]*full_rect[1][1]
                self.full_template_area = self.full_template.shape[0]*self.full_template.shape[1]
                full_box = cv2.boxPoints(full_rect).astype(int)
                inside = 0

                for corner in full_box:
                    if (corner[0] > 0 and corner[0] < image.shape[1]) and (corner[1] > 0 and corner[1] < image.shape[0]) :
                        inside += 1

                if inside > 1:
                    if full_rect_area > self.full_template_area*0.7:
                        FULL_DETECTED = True
                else:
                    CORRECT = "full_rect"
                    print("Invalid full detection: {} inside image".format(inside))
            if FULL_DETECTED:
                if full_rect:
                    full_box = cv2.boxPoints(full_rect).astype(int)
                    cv2.drawContours(image, [full_box], 0, 255, 2)
                    # full_croppedRotated = crop_rotate(image, full_rect, 0)
                    full_croppedRotated = self.crop_rotate(image, full_rect)

                    ### BEE
                    bee_rect = self.sift(BEE_TEMPLATE, full_croppedRotated)
                    if bee_rect == None: 
                        CORRECT = "bee_rect"
                        print("NO BEE -> Exception")
                    else:
                        BEE_DETECTED = True
                        bee_box = cv2.boxPoints(bee_rect).astype(int)

                        ### CAT
                        cat_rect = self.sift(CAT_TEMPLATE, full_croppedRotated)
                        if cat_rect == None: 
                            print("NO BEE -> Exception")
                            CORRECT = "bee_rect & cat_rect"
                        else:
                            CAT_DETECTED = True
                            cat_box = cv2.boxPoints(cat_rect).astype(int)

                    if BEE_DETECTED and CAT_DETECTED:
                        # cropped_center = (int(full_croppedRotated.shape[1]/2), int(full_croppedRotated.shape[0]/2))
                        bee_center = (int(bee_rect[0][0]), int(bee_rect[0][1]))
                        cat_center = (int(cat_rect[0][0]), int(cat_rect[0][1]))
                        
                        crayon_point_top_left   = self.create_point_from_origin(center=bee_center, standard=cat_center, dist=70, angle=0)
                        crayon_point_bot_right  = self.create_point_from_origin(center=bee_center, standard=cat_center, dist=155, angle=-80)

                        crayon_cropped = full_croppedRotated[crayon_point_top_left[1]:crayon_point_bot_right[1], crayon_point_top_left[0]:crayon_point_bot_right[0]]

                        crayon_rect = self.sift(self.crayon_template, crayon_cropped)
                        if crayon_rect == None: 
                            print("NO CRAYON -> Exception")
                            CORRECT = "crayon_rect"

                        else:
                            crayon_color_cropped = self.crop_rotate(crayon_cropped, crayon_rect)


                            if not isinstance(crayon_color_cropped,type(None)):
                                metric_val_list = self.compare_2_images_hsv(self.crayon_template, crayon_color_cropped)
                                jugdement_value = np.min(metric_val_list)

                                print("     metric_val_list min {} | max: {}".format(jugdement_value, np.max(metric_val_list)))
                                CRAYON_DETECTED = True
                                # if jugdement_value > 20: 
                                #     DETECTED_PERCENTS.append(percent)
                                if jugdement_value > 0.1:
                                    CRAYON_DETECTED = True
                                    CORRECT = True
                                else:
                                    print(" FAILED comparison: {}".format(jugdement_value))
                                
                                image[int(image.shape[0]-crayon_color_cropped.shape[0]-self.crayon_template.shape[0]-10):image.shape[0]-self.crayon_template.shape[0]-10,
                                    int(image.shape[1]/2-crayon_color_cropped.shape[1]/2):int(image.shape[1]/2+crayon_color_cropped.shape[1]/2)] = crayon_color_cropped

                            else:
                                # CRAYON_DETECTED = True

                                CORRECT = "Empty_cropped"
                                print("Empty cropped")
                            
                        # if CRAYON_DETECTED:
                        # _crayon_area = crayon_rect[1][0]*crayon_rect[1][1]

                        crayon_box = cv2.boxPoints(crayon_rect).astype(int)
                        cv2.drawContours(crayon_cropped, [crayon_box], 0, 150, 2)

                        # image[0:0+crayon_cropped.shape[0], 100:100+crayon_cropped.shape[1] ] = crayon_cropped   

                        # image[int(image.shape[0]-crayon_cropped.shape[0]-self.crayon_template.shape[0]):image.shape[0]-self.crayon_template.shape[0],
                        #       int(image.shape[1]/2-crayon_cropped.shape[1]/2):int(image.shape[1]/2+crayon_cropped.shape[1]/2)] = crayon_cropped

                        cv2.circle(full_croppedRotated, crayon_point_top_left, 5, (255, 255, 255), thickness=-1, lineType=8, shift=0)
                        cv2.circle(full_croppedRotated, crayon_point_bot_right, 5, (255, 255, 255), thickness=-1, lineType=8, shift=0)

                        cv2.circle(full_croppedRotated, cat_center, 5, (0, 255, 255), thickness=-1, lineType=8, shift=0)
                        cv2.circle(full_croppedRotated, bee_center, 5, (255, 255, 0), thickness=-1, lineType=8, shift=0)
                        cv2.line(full_croppedRotated, bee_center, cat_center, (0, 255, 0), thickness=3) 
                        cv2.line(full_croppedRotated, bee_center, crayon_point_top_left, (0, 255, 0), thickness=3) 
                        cv2.line(full_croppedRotated, bee_center, crayon_point_bot_right, (0, 255, 0), thickness=3) 
                            
                        cv2.drawContours(full_croppedRotated, [bee_box], 0, 255, 2)
                        cv2.drawContours(full_croppedRotated, [cat_box], 0, 150, 2)
                        cv2.drawContours(image, [full_box], 0, 255, 2)
                        
                        image[0:0+full_croppedRotated.shape[0], 0:0+full_croppedRotated.shape[1] ] = full_croppedRotated   

                    else:
                        print("CANNOT DETECT BEE OR CAT") 
                
                else:
                    print("NO BOOK")

        except Exception as e:
            traceback.print_exc() 
            cv2.putText(image, "Cannot detect", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
        
        return image, CORRECT, jugdement_value

    def programA(self, image):
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        for step in load_prog['steps']:
            if step['stepId'] == 1:
                convey = step['data']

        FULL_DETECTED = False
        CRAYON_DETECTED = False
        percent = 0
        jugdement_value = 0

        CORRECT = False
        image = image[convey[1]:convey[3], :]
        try:
            ### ORB
            full_rect = self.orb(self.full_template, image)
            if full_rect == None: 
                CORRECT = "full_rect"
                print("NO BOOK -> Exception")
            else:
                full_rect_area = full_rect[1][0]*full_rect[1][1]
                self.full_template_area = self.full_template.shape[0]*self.full_template.shape[1]
                full_box = cv2.boxPoints(full_rect).astype(int)
                inside = 0

                for corner in full_box:
                    if (corner[0] > 0 and corner[0] < image.shape[1]) and (corner[1] > 0 and corner[1] < image.shape[0]) :
                        inside += 1

                if inside > 1:
                    if full_rect_area > self.full_template_area*0.7:
                        FULL_DETECTED = True
                else:
                    CORRECT = "full_rect"
                    print("Invalid full detection: {} inside image".format(inside))

            if FULL_DETECTED:
                if full_rect:
                    full_box = cv2.boxPoints(full_rect).astype(int)
                    full_croppedRotated = self.crop_rotate(image, full_rect)
                    crayon_rect = self.sift(self.crayon_template, full_croppedRotated)
                    if crayon_rect == None: 
                        print("NO CRAYON -> Exception")
                        CORRECT = "crayon_rect"

                    else:
                        crayon_color_cropped = self.crop_rotate(full_croppedRotated, crayon_rect)

                        if not isinstance(crayon_color_cropped,type(None)):
                            metric_val_list = self.compare_2_images_hsv(self.crayon_template, crayon_color_cropped)
                            jugdement_value = np.min(metric_val_list)

                            print("     metric_val_list min {} | max: {}".format(jugdement_value, np.max(metric_val_list)))
                            CRAYON_DETECTED = True
                            # if jugdement_value > 20: 
                            #     DETECTED_PERCENTS.append(percent)
                            if jugdement_value > 0.1:
                                CRAYON_DETECTED = True
                                CORRECT = True
                            else:
                                print(" FAILED comparison: {}".format(jugdement_value))
                            
                            # image[int(image.shape[0]-crayon_color_cropped.shape[0]-self.crayon_template.shape[0]-10):image.shape[0]-self.crayon_template.shape[0]-10,
                            #         int(image.shape[1]/2-crayon_color_cropped.shape[1]/2):int(image.shape[1]/2+crayon_color_cropped.shape[1]/2)] = crayon_color_cropped

                        else:
                            # CRAYON_DETECTED = True

                            CORRECT = "Empty_cropped"
                            print("Empty cropped")
                    if self.show:
                        # if CRAYON_DETECTED:
                        # _crayon_area = crayon_rect[1][0]*crayon_rect[1][1]
                        cv2.drawContours(image, [full_box], 0, 255, 2)

                        crayon_box = cv2.boxPoints(crayon_rect).astype(int)
                        cv2.drawContours(full_croppedRotated, [crayon_box], 0, 150, 2)

                        # image[0:0+crayon_cropped.shape[0], 100:100+crayon_cropped.shape[1] ] = crayon_cropped   

                        # image[int(image.shape[0]-crayon_cropped.shape[0]-self.crayon_template.shape[0]):image.shape[0]-self.crayon_template.shape[0],
                        #       int(image.shape[1]/2-crayon_cropped.shape[1]/2):int(image.shape[1]/2+crayon_cropped.shape[1]/2)] = crayon_cropped

                        image[0:0+full_croppedRotated.shape[0], 0:0+full_croppedRotated.shape[1] ] = full_croppedRotated   

                        image[int(image.shape[0]-self.crayon_template.shape[0]):image.shape[0],
                                int(image.shape[1]/2-self.crayon_template.shape[1]/2):int(image.shape[1]/2+self.crayon_template.shape[1]/2)] = self.crayon_template
            
            else:
                print("NO BOOK")

        except Exception as e:
            traceback.print_exc() 
            cv2.putText(image, "Cannot detect", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
        
        return image, CORRECT, jugdement_value

    def programUPG(self, image):
        # gpu_image = cv2.cuda_GpuMat()
        # gpu_image.upload(image)
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        FULL_DETECTED = False
        for step in load_prog['steps']:
            if step['stepId'] == 1:
                origin = step['data']
            elif step['stepId'] == 2:
                areas = step['data']
        try:
            full_rect = self.orb_gpu(self.full_template, image)
            print("orb_gpu full_rect", full_rect)
            if full_rect == None: 
                CORRECT = "full_rect"
                print("NO BOOK -> Exception")
            else:
                full_rect_area = full_rect[1][0]*full_rect[1][1]
                self.full_template_area = self.full_template.shape[0]*self.full_template.shape[1]
                full_box = cv2.boxPoints(full_rect).astype(int)
                inside = 0

                for corner in full_box:
                    if (corner[0] > 0 and corner[0] < image.shape[1]) and (corner[1] > 0 and corner[1] < image.shape[0]) :
                        inside += 1

                if inside > 1:
                    if full_rect_area > self.full_template_area*0.7:
                        FULL_DETECTED = True
                else:
                    CORRECT = "full_rect"
                    print("Invalid full detection: {} inside image".format(inside))

            if FULL_DETECTED:
                if full_rect:
                    full_box = cv2.boxPoints(full_rect).astype(int)
                    full_croppedRotated = self.crop_rotate(image, full_rect)

                    if self.full_template.shape[0] > self.full_template.shape[1]:
                        w = min(full_rect[1])
                        h = max(full_rect[1])
                    else:
                        w = max(full_rect[1])
                        h = min(full_rect[1])

                    if 1:
                        cv2.drawContours(image, [full_box], 0, 255, 2)
                        for area in areas:
                            image = cv2.rectangle(image, (area[0]-origin[0]+int(full_rect[0][0]-w/2), area[1]-origin[1]+int(full_rect[0][1]-h/2)), 
                                                  (area[2]-origin[0]+int(full_rect[0][0]-w/2), area[3]-origin[1]+int(full_rect[0][1]-h/2)), (0, 255, 0), 2) 
            else:
                print("NO BOOK")

        except Exception as e:
            traceback.print_exc() 
            cv2.putText(image, "Cannot detect", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
        
        return image

    def export_frames(video_dir):
        success = 1
        vidObj = cv2.VideoCapture(video_dir) 
        count = 0
        while success: 
            success, image = vidObj.read() 
            if image is not None:
                if ROTATE:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cv2.imwrite(os.path.join(FRAMES_DIR, "frame%d.jpg" % count), image)
                count += 1

# checks whether frames were extracted 
success = 1

img_array = []
pro_w = 0
pro_h = 0

EXPORT_FRAMES = True
# BEE_TEMPLATES_DIR = '/home/acd/templateMatching/images/beeTemplate'
# self.crayon_templateS_DIR = '/home/acd/templateMatching/images/crayonTemplate'
# PROJECT_DIR = '/home/acd/templateMatching'

# TEMPLATE_DIR = BEE_TEMPLATES_DIR

# template = cv2.imread(os.path.join(BEE_TEMPLATES_DIR, "beeVideo.jpg"))
# img = cv2.imread(os.path.join(PROJECT_DIR, 'frame113.jpg'),cv2.COLOR_BGR2GRAY) # trainImage
# img3 = orb(template, img)
# plt.imshow(img3,)
# plt.show()
ROTATE = False

# GS_FRAMES_DIR = "/home/acd/templateMatching/gs/frames"
# GS_DIR = "/home/acd/templateMatching/gs"
# GS_TEMPLATE_DIR = "/home/acd/templateMatching/gs/template"

# BEE_TEMPLATE = cv2.imread(os.path.join(GS_TEMPLATE_DIR, "bee4.jpg"))
# self.full_template = cv2.imread(os.path.join(GS_TEMPLATE_DIR, "full5.jpg"))
# self.crayon_template = cv2.imread(os.path.join(GS_TEMPLATE_DIR, "crayon5.jpg"))
# CRAYON_REF_AREA = self.crayon_template.shape[0]*self.crayon_template.shape[1]
# CAT_TEMPLATE = cv2.imread(os.path.join(GS_TEMPLATE_DIR, "cat4.jpg"))


# VIDEO_DIR = GS_DIR + '/video'
# IMAGES_DIR = GS_DIR + '/images'
# FRAMES_DIR = GS_DIR + '/frames'

CONVEY_AREA_X_RANGE = [250, 1000]
CONVEY_AREA_Y_RANGE = [0, 700]

COMPARE_PERCENTAGE = 75
DETECTED_PERCENTS = []

source_image = cv2.imread(os.path.join("/home/lionel/templateMatching/editor/programs/1", "train_image.jpg"))
# source_image = source_image[100:500, 0:0+source_image.shape[1] ]
# cv2.imwrite(os.path.join(IMAGES_DIR, 'source_image.jpg'), source_image)
ROTATE = False
start = time.time()
vp = VideoProcessingGPU(False, "/home/lionel/templateMatching/editor/programs/1")
out = vp.programUPG(source_image)
# print("FPS: {}".format(round(1/(time.time()-start),3)))
# cv2.imwrite(os.path.join(IMAGES_DIR, 'output.jpg'), out)

# SOURCE_VIDEO_DIR = "/home/acd/templateMatching/gs/video/video.h264"
# # SOURCE_VIDEO_DIR = "/home/acd/templateMatching/gs/video/video_incorrect_3.h264"

# export_frames(SOURCE_VIDEO_DIR)

# vidObj = cv2.VideoCapture(SOURCE_VIDEO_DIR) 
# count       = 0
# TOTAL_QUAN_FOR_JUDGEMENT = 5
# DETECTION_LIST = [False] * TOTAL_QUAN_FOR_JUDGEMENT
# ACCEPTANCE_VALUE = TOTAL_QUAN_FOR_JUDGEMENT - 2

# MAX_PERCENT = 0
# EVER_DETECTED = False
# EVER_DETECTED_LIST = []
# EVER_DETECTED_MAX_ACCEPTANCE = 0

# while success: 
#     success, image = vidObj.read() 

#     if image is not None:

#         start_time = time.time()
#         image, ret, percent = process_simple(image)
#         if percent > MAX_PERCENT: MAX_PERCENT = percent
#         time_consumed   = time.time() - start_time
#         fps             = 1/time_consumed

#         if pro_w == 0:
#             pro_h, pro_w, layers = image.shape

#         count += 1

#         DETECTION_LIST.insert(len(DETECTION_LIST) - 1, DETECTION_LIST.pop(0))
#         if ret == True:
#             DETECTION_LIST[-1] = True
#         else:
#             DETECTION_LIST[-1] = False
#         print("Frame: {} | Last {}: {}".format(count, TOTAL_QUAN_FOR_JUDGEMENT, DETECTION_LIST.count(True)))
#         if DETECTION_LIST.count(True) > EVER_DETECTED_MAX_ACCEPTANCE: EVER_DETECTED_MAX_ACCEPTANCE = DETECTION_LIST.count(True)
#         if DETECTION_LIST.count(True) >= ACCEPTANCE_VALUE:
#             color = (0, 255, 0)
#             EVER_DETECTED = True
#             EVER_DETECTED_LIST.append(count)
#             print("DETECTED")
#         else:
#             color = (255,0,0)
#         cv2.putText(image, "FRAME: {} | fps: {} | Detection: {}/{}".format(count, round(fps, 2), DETECTION_LIST.count(True), TOTAL_QUAN_FOR_JUDGEMENT), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)
#         cv2.putText(image, str(ret), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, 2)
#         cv2.putText(image, str(percent), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)
#         cv2.putText(image, "EVER_DETECTED: {} @ frame {}".format(EVER_DETECTED, EVER_DETECTED_LIST), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, 2)
        
#         img_array.append(image)

# print("MAX_PERCENT", MAX_PERCENT)
# print("EVER_DETECTED_MAX_ACCEPTANCE", EVER_DETECTED_MAX_ACCEPTANCE)
# print("EVER DETECTED", EVER_DETECTED)
# if DETECTED_PERCENTS != []:
#     print("Average DETECTED_PERCENTS: {} | Quan: {}".format(round(np.mean(DETECTED_PERCENTS),2), len(DETECTED_PERCENTS)))
# # choose codec according to format needed
# fourcc = cv2.VideoWriter_fourcc(*'DIVX') 

# video = cv2.VideoWriter(os.path.join(VIDEO_DIR, 'detected_video.avi'), fourcc, 15, (pro_w, pro_h))

# for j in range(len(img_array)):
#     video.write(img_array[j])

# cv2.destroyAllWindows()
# video.release()