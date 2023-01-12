import cv2 as cv
import numpy as np

RIGHT = 2
LEFT = 1

def slope(x, y, w, z, frame):
    return (y - z) / (x - w)


class LaneDetector():
    def __init__(self):
        self.frame_processed = 0
        #                           0 = r_low , 1 = r_up , 2 = l_low , 3 = l_up
        self.prev_lines_coords = [(900, 625), (655, 415), (223, 415), (537, 625)]
        # range of right_lower_x values representing static driving
        self.right_lane_anchor_TH = (950, 1170)
        # frame counter for printing message
        self.message_frame_counter = 0
        # 0 = false / 1 = to the LEFT / 2 = to the RIGHT                         
        self.lane_switching = 0

    def drawLane(self, lines, frame):
        # getting lane coordinates
        # every ~0.5 a sec choose new lines (videowriter set to 25fps)
        if self.frame_processed % 11 == 1:
            self.prev_lines_coords = self.chooseTwoLines(lines)
            # print(self.frame_processed)
        right_lower, right_upper, left_lower, left_upper = self.prev_lines_coords

        mask = np.zeros_like(frame)
        # lane mask
        cv.fillPoly(mask, np.array([[left_lower, left_upper, right_upper, right_lower]], dtype=np.int32),
                    (255, 255, 255))
        # making the road between lines darker
        overlay = frame.copy()
        overlay[mask != 0] = 40
        # making the highlighted lanes with opacity
        alpha = 0.4
        frame = cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv.line(frame, right_lower, right_upper, (0, 0, 255), thickness=3)
        cv.line(frame, left_lower, left_upper, (0, 0, 255), thickness=3)

        return frame

    def getMaskVertices(self,processed_frame):
        # generating vertices according to current lane switching status
        if self.lane_switching == 0:
            return np.array([[
                              (125, processed_frame.shape[0] - 100),
                              (550, 430), 
                              (750, 430),
                              (processed_frame.shape[1] - 125, processed_frame.shape[0] - 100)]],
                            dtype=np.int32)
        elif self.lane_switching == LEFT:
            return np.array([[
                              (50, processed_frame.shape[0] - 100), 
                              (50, 430),
                              (750, 430),
                              (processed_frame.shape[1] - 125, processed_frame.shape[0] - 100)]],
                            dtype=np.int32)
        else:
            return np.array([[
                              (125, processed_frame.shape[0] - 100),
                              (550, 430), 
                              (processed_frame.shape[1] - 50, 430),
                              (processed_frame.shape[1] - 50, processed_frame.shape[0] - 100)]],
                            dtype=np.int32)
     

    def createMask(self, processed_frame):
        # create mask according to situation
        vertices = self.getMaskVertices(processed_frame)

        mask = np.zeros_like(processed_frame)
        # plot the trapeze on the mask
        cv.fillPoly(mask, vertices, (255, 255, 255))

        return cv.bitwise_and(processed_frame, mask)

    def laneSwitchDetection(self, frame):
        cur_right_low = self.prev_lines_coords[0][0]

        if cur_right_low in range(self.right_lane_anchor_TH[0],self.right_lane_anchor_TH[1]):
            self.lane_switching = 0
            return frame
        
        self.message_frame_counter += 1
        message = "Changing lanes to the "

        if self.message_frame_counter <= 80:

            # changing lane to the RIGHT
            if cur_right_low < self.right_lane_anchor_TH[0]:
                message += "RIGHT!"
                self.lane_switching = RIGHT

            elif cur_right_low > self.right_lane_anchor_TH[1]:
                message += "LEFT!"
                self.lane_switching = LEFT
                

            frame = cv.putText(frame, message, (70,70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)

        else:
            self.message_frame_counter = 0
            self.lane_switching = 0

        return frame

    def yellowAndwhiteMask(self, frame):
        yellow_th = [np.array([15, 50, 90]), np.array([25, 255, 255])]
        sens = 40
        white_th = [np.array([0, 0, 255 - sens]), np.array([255, sens, 255])]

        yellow_mask = cv.inRange(frame, yellow_th[0], yellow_th[1])
        white_mask = cv.inRange(frame, white_th[0], white_th[1])
        white_mask = cv.dilate(white_mask, np.ones((5, 2), dtype=np.uint8))
        return cv.bitwise_or(yellow_mask, white_mask)

    def preprocess(self, frame):

        frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # get the relevant crop of the frame
        trapeze_frame = self.createMask(frame_HSV)
        # cv.imshow('tmuna', trapeze_frame)

        # set white and yellow TH for lane recognition
        trapeze_frame = cv.cvtColor(cv.cvtColor(trapeze_frame, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

        # cropped trapeze with white and yellow mask
        masked_frame = cv.bitwise_and(trapeze_frame, self.yellowAndwhiteMask(frame_HSV))

        # noise cancellation
        masked_frame = cv.GaussianBlur(masked_frame, (7, 7), cv.BORDER_WRAP)
        # edge detection
        return cv.Canny(masked_frame, 50, 150)

    def detect(self, processed_frame, org_frame):  # return left lane mask and right lane mask
        self.frame_processed += 1
        # get all lines in frame
        lines = cv.HoughLinesP(processed_frame, rho=3, theta=np.pi / 160, threshold=40,minLineLength=130, maxLineGap=180)
        frame_with_lanes = self.drawLane(lines, org_frame)
        return self.laneSwitchDetection(frame_with_lanes)

    def chooseTwoLines(self, lines):

        right_x_lower = [0]
        right_x_upper = [0]
        left_x_lower = [0]
        left_x_upper = [0]

        # absolute upper and lower border of the lane line we wish to draw
        y_lower = 625
        y_upper = 415

        if lines is None:
            return self.prev_lines_coords

        for line in lines:
            x1, y1, x2, y2 = line[0]

            line_angle = slope(x1, y1, x2, y2, self.frame_processed)
            b = y1 - line_angle * x1

            if self.lane_switching == 0:
                if 0.33 < line_angle < 1.73:  # arctan(1.73) = 60 degrees, arctan(0.33) = 20 degrees
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.45:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

            elif self.lane_switching == LEFT:
                if 0.33 < line_angle < 1.73:  # arctan(1.1) = 48 degrees, arctan(0.6) = 30 degrees
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.15:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

            else:
                if 0.15 < line_angle < 1:  # arctan(1.73) = 60 degrees, arctan(0.33) = 20 degrees
                    right_x_lower.append((y_lower - b) / line_angle)
                    right_x_upper.append((y_upper - b) / line_angle)
                elif -1. < line_angle < -0.45:
                    left_x_lower.append((y_lower - b) / line_angle)
                    left_x_upper.append((y_upper - b) / line_angle)

        # 0 = r_low , 1 = r_up , 2 = l_low , 3 = l_up
        coordinates = [self.createSafeCoords(np.median(right_x_lower), y_lower, 0), \
                       self.createSafeCoords(np.median(right_x_upper), y_upper, 1), \
                       self.createSafeCoords(np.median(left_x_lower), y_lower, 2), \
                       self.createSafeCoords(np.median(left_x_upper), y_upper, 3)]

        return coordinates

    def createSafeCoords(self, xval, yval, idx):
        if xval < 1 or (np.abs(xval - self.prev_lines_coords[idx][0]) > 150):
            return self.prev_lines_coords[idx]

        return (int(xval), yval)


LD = LaneDetector()

stream = cv.VideoCapture(f"C:\\Users\\a\\Documents\\Computer Science\\AI Is Math\\proj\\cropped_sample2.mp4")

ret, frame = stream.read()

out = cv.VideoWriter('when I switch lanes.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                     (frame.shape[1], frame.shape[0]))

if stream.isOpened() == False:
    print('Couldn\'t load file "sample.mp4"')
    exit(-1)

while stream.isOpened():
    ret, frame = stream.read()
    if ret:
        processed_frame = LD.preprocess(frame)
        drawn_frame = LD.detect(processed_frame, frame)
        cv.imshow('tmuna', drawn_frame)
        cv.waitKey(5)
        out.write(drawn_frame)
    else:
        break

stream.release()
cv.destroyAllWindows()

