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
# ### ORB
### Source: https://github.com/armaanpriyadarshan/Rotation-Tracking-OpenCV/blob/main/main.py
# template_img_bw = cv2.imread(os.path.join(BEE_TEMPLATES_DIR, 'beeVideo2.jpg'), cv2.COLOR_BGR2GRAY) # queryImage
# train_img_bw = cv2.imread(os.path.join(PROJECT_DIR, 'frame113.jpg'),cv2.COLOR_BGR2GRAY) # trainImage

class VideoProcessing():
    def __init__(self, show=False, program_dir=None):
        self.program_dir = program_dir
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        if load_prog['typeId'] == 'A':
            self.full_template = cv2.imread(os.path.join(program_dir, "templates/full.jpg"))
            self.crayon_template = cv2.imread(os.path.join(program_dir, "templates/crayon.jpg"))
        elif load_prog['typeId'] == 'UPG':
            self.full_template = cv2.imread(os.path.join(program_dir, "templates/10ports.jpg"))
        elif load_prog['typeId'] == 'B':
            self.full_template = cv2.imread(os.path.join(program_dir, "templates/crayon.jpg"))

        self.show = show
        self.start_point = None
        self.init_counter_variables()

    def orb(self, template_img, original_img):
        try:
            h, w, layers = template_img.shape
            template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY) 
            img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY) 
            
            template_gray = cv2.medianBlur(template_gray, 5)
            img_gray = cv2.medianBlur(img_gray, 5)

            orb = cv2.ORB_create() 
            
            queryKeypoints, queryDescriptors = orb.detectAndCompute(template_gray, None) 
            trainKeypoints, trainDescriptors = orb.detectAndCompute(img_gray, None) 
            
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
            
            # Initiate SIFT detector
            sift = cv2.xfeatures2d.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(template_gray,None)
            kp2, des2 = sift.detectAndCompute(img_gray,None)


            # Initialize the Matcher for matching 
            # the keypoints and then match the 
            # keypoints 
            matcher = cv2.BFMatcher() 
            matches = matcher.match(des1, des2) 

            matches = sorted(matches, key=lambda x: x.distance)

            template_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            img_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(template_pts, img_pts, cv2.RANSAC)

            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            rect = cv2.minAreaRect(transformed_corners)

            return rect
        except:
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

    def detect_crayon_colors_hsv(self, image, areas, dir, origin):
        status = None
        if dir == 'Vertical':
            sorted_ref_areas = sorted(areas, key=lambda x: (x[0]+x[2]))
        elif dir == 'Horizontal':
            sorted_ref_areas = sorted(areas, key=lambda x: (x[1]+x[3]))
        # Convert the image from BGR to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        template_gray = cv2.cvtColor(self.full_template, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cx, cy = 0, 0  
        detected_contours = []
        for idx, area in enumerate(sorted_ref_areas):
        # for color, (lower, upper, draw_color) in color_ranges.items():
            # get template histogram
            template_color_region_gray = template_gray[area[1]-origin[1]:area[3]-origin[1], area[0]-origin[0]:area[2]-origin[0]]
            hist_gray_template = cv2.calcHist([template_color_region_gray], [0], None, [256], [0, 256])
            hist_gray_template /= hist_gray_template.sum() # Normalize histograms

            lower_np = np.array((area[4], area[6], area[8]), dtype=np.uint8)
            upper_np = np.array((area[5], area[7], area[9]), dtype=np.uint8)

            mask = cv2.inRange(hsv_image, lower_np, upper_np)
            cv2.bitwise_and(image, image, mask=mask)

            # Find contours in the result image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

                # Get the top 2 largest contours
                top_contours = sorted_contours[:2]
                max_correlation = -10 # because the range is -1 to 1
                max_correlation_idx = None
                for i_, contour_ in enumerate(top_contours):
                    mask = np.zeros_like(hsv_image[:,:,0])
                    cv2.drawContours(mask, [contour_], 0, (255), -1)

                    # Extract the contour region from the HSV image using the mask
                    contour_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)

                    # Calculate the histogram of the grayscale channel
                    hist_gray = cv2.calcHist([contour_gray], [0], None, [256], [0, 256])

                    # Normalize the histogram
                    hist_gray /= hist_gray.sum()
                    correlation = cv2.compareHist(hist_gray_template, hist_gray, cv2.HISTCMP_CORREL)
                    if correlation > max_correlation: 
                        max_correlation = correlation
                        max_correlation_idx = i_
                    # print(idx, i_, max_correlation_idx, correlation, cv2.contourArea(contour_))

                largest_contour = max(contours, key=cv2.contourArea)
                best_contour = top_contours[max_correlation_idx]
                moments = cv2.moments(best_contour)

                # Calculate centroid
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

                    # Draw a small circle at the centroid
                    cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(image, str(idx+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                detected_contours.append([idx, cv2.contourArea(best_contour), cx, cy])

            else:
                detected_contours.append(None)
        if None in detected_contours:
            status = "Some color cannot be detected"
            print(status)
            return image, status

        if len(detected_contours) != len(areas):
            status = "Wrong quantity detected: detected {} vs. desire {}".format(len(detected_contours), len(areas))
            return image, status

        # Validate contours
        max_area = 0
        min_area = 0
        for i, cnt in enumerate(detected_contours):
            _area = cnt[1]
            if i == 0: min_area = _area
            if _area > max_area: max_area = _area
            if _area < min_area: min_area = _area
        print("Min/Max area", min_area/max_area)
        # valid_area_ratio = False
        # if min_area > 0.7* max_area:
        #     valid_area_ratio = True
        # else:
        #     status = "Invalid Area Ratio: {}".format(min_area/max_area)
        #     return image, status

        if 1: #valid_area_ratio:
            ascending_ratio_sequence = True
            if dir == 'Vertical':
                sorted_detected_contours = sorted(detected_contours, key=lambda x: x[2])
            elif dir == 'Horizontal':
                sorted_detected_contours = sorted(detected_contours, key=lambda x: x[3])
            for i in range(len(sorted_detected_contours) - 1):
                if sorted_detected_contours[i][0] > sorted_detected_contours[i + 1][0]:
                    ascending_ratio_sequence = False
                    status = "Wrong Sequence"
                    break

        if ascending_ratio_sequence:
            status = True

        return image, status
    
    def compare_2_images_hsv(self, image1, image2, quantity, dir):
        # cv2.imwrite(os.path.join(self.program_dir, "compare_1.jpg"), image1)
        # cv2.imwrite(os.path.join(self.program_dir, "compare_2.jpg"), image2)

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
        if dir == 'Vertical':
            step        = int(w/quantity)
            step_dis    = int(step/2)
        elif dir == 'Horizontal':
            step        = int(h/quantity)
            step_dis    = int(step/2)
        dis = 0
        metric_val_list = []
        for i in range(2*quantity-1):
            if dir == 'Vertical':
                image1_splitted = img1_hsv[0:h, dis:dis+step]
                # cv2.imwrite(os.path.join(self.program_dir, "image1_splitted_{}.jpg".format(i)), image1_splitted)
                
                image2_splitted = img2_hsv[0:h, dis:dis+step]
                # cv2.imwrite(os.path.join(self.program_dir, "image2_splitted_{}.jpg".format(i)), image2_splitted)
            elif dir == 'Horizontal':
                image1_splitted = img1_hsv[dis:dis+step, 0:w]
                # cv2.imwrite(os.path.join(self.program_dir, "image1_splitted_{}.jpg".format(i)), image1_splitted)
                
                image2_splitted = img2_hsv[dis:dis+step, 0:w]
                # cv2.imwrite(os.path.join(self.program_dir, "image2_splitted_{}.jpg".format(i)), image2_splitted)
            
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
        for entry in load_prog['entries']:
            if entry['name'] == 'Crayon Quantity':
                crayon_quantity = int(entry['data'])
        for item in load_prog['combobox']:
            if item['name'] == 'Choose crayon direction':
                crayon_dir = item['data']

        FULL_DETECTED = False
        CRAYON_DETECTED = False
        percent = 0
        jugdement_value = 0

        CORRECT = False
        image = image[convey[1]:convey[3], :]
        try:
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
                            metric_val_list = self.compare_2_images_hsv(self.crayon_template, crayon_color_cropped, crayon_quantity, crayon_dir)

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
                        full_croppedRotated = self.detect_crayon_colors_hsv(full_croppedRotated)


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
    
    def init_counter_variables(self):
        self.new_register = False
        self.left_flag = False
        self.passed_middle_line = False
        self.current_count_value = self.corrected_count_value = 0
        self.left_counter = 0
        self.need_to_detect_crayon = False
        self.current_is_correct = False
        self.warning_wrong = False

    def count(self, image, rect, detected):
        h, w, layers = image.shape
        if rect != None:
            # Access into first half
            if rect[0][0] < w/2 and (self.left_flag or self.current_count_value == 0):
                self.new_register = True
                self.left_flag = False
                self.passed_middle_line = False
                self.current_count_value += 1

                self.need_to_detect_crayon = True
                self.current_is_correct = False
                self.warning_wrong = False
            
            if self.new_register:
                if rect[0][0] > w/2: # passed the middle line
                    self.passed_middle_line = True
                    self.new_register = False

        elif rect == None or not detected:
            if (self.new_register or self.passed_middle_line) and self.left_counter < MAX_LEFT_COUNTER:
                self.left_counter += 1
            if self.left_counter == MAX_LEFT_COUNTER:
                self.left_flag = True
                # in 2nd half
                if self.passed_middle_line:
                    self.passed_middle_line = False   
                    if self.current_is_correct:
                        self.corrected_count_value += 1
                    else:
                        self.warning_wrong = True
                # or in the 1st half
                elif self.new_register:
                    self.new_register = False
                    if self.current_count_value > 0: 
                        self.current_count_value -= 1
            
        if rect != None:
            print(self.new_register, self.passed_middle_line, self.left_flag, detected, self.current_count_value, self.corrected_count_value, rect[0], w/2, rect[0][0] > w/2)
        else:
            print(self.new_register, self.passed_middle_line, self.left_flag, detected, self.current_count_value, self.corrected_count_value)

    def programB(self, image):
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        for step in load_prog['steps']:
            if step['stepId'] == 1:
                convey = step['data']
            elif step['stepId'] == 2:
                origin = step['data']
            elif step['stepId'] == 3:
                crayon_area = step['data']
            elif step['stepId'] == 4:
                color_areas = step['data']
        for entry in load_prog['entries']:
            if entry['name'] == 'Crayon Quantity':
                crayon_quantity = int(entry['data'])
        for item in load_prog['combobox']:
            if item['name'] == 'Choose crayon direction':
                crayon_dir = item['data']

        FULL_DETECTED = False
        CRAYON_DETECTED = False
        percent = 0
        jugdement_value = 0

        CORRECT = False
        image = image[convey[1]:convey[3], :]
        cv2.putText(image, "COUNTER: {}/{}".format(self.corrected_count_value, self.current_count_value), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)
        if self.current_is_correct:
            cv2.putText(image, "FOUND A CORRECT ONE", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)
        if self.warning_wrong:
            cv2.putText(image, "LAST ONE WAS CAUGHT WRONG", (50, image.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)

        try:
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
                print("detected crayon")
                cv2.drawContours(image, [full_box], 0, 255, 2)
                if self.current_is_correct:
                    if not self.passed_middle_line:
                        cv2.putText(image, "DETECTED -> Wait for passing mid line", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)
                    else:
                        cv2.putText(image, "DETECTED -> PASSED mid line", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)

                status = None
                if self.need_to_detect_crayon:
                    full_box = cv2.boxPoints(full_rect).astype(int)
                    full_croppedRotated = self.crop_rotate(image, full_rect)
                    if self.full_template.shape[0] > self.full_template.shape[1]:
                        w = min(full_rect[1])
                        h = max(full_rect[1])
                    else:
                        w = max(full_rect[1])
                        h = min(full_rect[1])
                    crayon_region = full_croppedRotated[crayon_area[1]-origin[1]:crayon_area[3]-origin[1],
                                        crayon_area[0]-origin[0]:crayon_area[2]-origin[0]]
                    crayon_region, status = self.detect_crayon_colors_hsv(crayon_region, color_areas, crayon_dir, origin)

                    if status:
                        if status == True:
                            self.need_to_detect_crayon = False
                            self.current_is_correct = True
                            print("Correct! Wait for passing middle line...")
                            cv2.putText(image, "CORRECT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, 2)
                        elif status != None:
                            cv2.putText(image, status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 2)

                        image[image.shape[0]-crayon_region.shape[0]:image.shape[0], 0:0+crayon_region.shape[1] ] = crayon_region
            else:
                full_rect = None
                print("NO BOOK")
                cv2.putText(image, "Waiting for a new one", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, 2)

            self.count(image, full_rect, FULL_DETECTED)
            cv2.line(image, (int(image.shape[1]/2), 0), (int(image.shape[1]/2), int(image.shape[0])), (0,0,255), 5)

        except Exception as e:
            traceback.print_exc() 
            cv2.putText(image, "Cannot detect", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, 2)
        
        return image, CORRECT, jugdement_value

    def programUPG(self, image):
        FULL_DETECTED = False
        prog_data_file = os.path.join(self.program_dir, DATA_INFO_FILENAME)
        load_prog = json.load(open(prog_data_file))
        for step in load_prog['steps']:
            if step['stepId'] == 1:
                convey = step['data']
            elif step['stepId'] == 2:
                origin = step['data']
            elif step['stepId'] == 3:
                areas = step['data']
        try:
            image = image[convey[1]:convey[3], :]
            full_rect = self.sift(self.full_template, image)
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

# EXPORT_FRAMES = True
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

# CONVEY_AREA_X_RANGE = [250, 1000]
# CONVEY_AREA_Y_RANGE = [0, 700]

# COMPARE_PERCENTAGE = 75
# DETECTED_PERCENTS = []

# source_image = cv2.imread(os.path.join(FRAMES_DIR, "frame_20.jpg"))
# # source_image = source_image[100:500, 0:0+source_image.shape[1] ]
# cv2.imwrite(os.path.join(IMAGES_DIR, 'source_image.jpg'), source_image)
# ROTATE = False
# start = time.time()
# vp = VideoProcessing()
# out, ret, percent = vp.process_simple(source_image, show=True)
# print("FPS: {}".format(round(1/(time.time()-start),3)))
# cv2.imwrite(os.path.join(IMAGES_DIR, 'output.jpg'), out)

# source_image = cv2.imread(os.path.join("/home/lionel/templateMatching/editor/programs/7", "train_image.jpg"))
# # source_image = source_image[100:500, 0:0+source_image.shape[1] ]
# # cv2.imwrite(os.path.join(IMAGES_DIR, 'source_image.jpg'), source_image)
# ROTATE = False
# start = time.time()
# vp = VideoProcessing(True, "/home/lionel/templateMatching/editor/programs/7")
# vp.need_to_detect_crayon = True
# out = vp.programB(source_image)
# cv2.imwrite(os.path.join("/home/lionel/templateMatching/editor/programs/7", 'output.jpg'), out[0])

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