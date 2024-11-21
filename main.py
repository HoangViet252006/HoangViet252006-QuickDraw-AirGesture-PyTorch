import os
from handTracker import *
import cv2
import numpy as np
from inferences import inference


WHITE_COLOUR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
BLUE_COLOR = (255, 0, 0)

class ColorRect():
    def __init__(self, x, y, w, h, color, text = '', alpha = 0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        #draw the box
        alpha = self.alpha
        bg_rec = img[self.y : self.y + self.h, self.x : self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 0)
        
        # Putting the image back to its position
        img[self.y : self.y + self.h, self.x : self.x + self.w] = res

        #put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x + self.w/2 - tetx_size[0][0]/2), int(self.y + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text, text_pos , fontFace, fontScale, text_color, thickness)


    def isHover(self,x,y):
        if (self.x + self.w > x > self.x) and (self.y + self.h> y >self.y):
            return True
        return False


def camera_app():
    #initilize categories
    subdir = [os.path.join("data", sub) for sub in os.listdir("data")]
    categories = [os.path.basename(sub) for sub in subdir]
    categories = [cate.split("_")[-1].replace(".npy", "") for cate in categories]
    #initilize the habe detector
    detector = HandTracker(detectionCon=0.8)

    #initilize the camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # creating canvas to draw on it
    canvas = np.zeros((720,1280,3), np.uint8)

    # define a previous point to be used with drawing a line
    px,py = 0,0
    color = (255,0,0)
    brushSize = 5
    eraserSize = 20



    colors = []
    #blue
    colors.append(ColorRect(800,0,100,100, BLUE_COLOR))
    #erase (black)
    colors.append(ColorRect(900,0,100,100, BLACK_COLOR, "Eraser"))
    #clear
    clear = ColorRect(1000,0,100,100, (100,100,100), "Clear")




    #define a white board to draw on
    whiteBoard = ColorRect(130, 120, 1020, 580, WHITE_COLOUR, alpha = 0.6)

    frames_pred = 10
    Counter_TextPredict = 0;
    pred  = ""
    while True:
        flag, frame = cap.read()
        if not flag:
            break

        if frames_pred > 0:
            frames_pred -=1
        else:
            frames_pred = 20


        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        ########## pen colors' boxes #########
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)


        detector.findHands(frame)
        positions = detector.getPostion(frame, draw=False)
        upFingers = detector.getUpFingers(frame)



        if upFingers:
            x, y = positions[8][0], positions[8][1]
            if upFingers[0] and not whiteBoard.isHover(x, y):
                px, py = 0, 0

                ####### chose a color for drawing #######
                for cb in colors:
                    if cb.isHover(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                    #Clear
                    if clear.isHover(x, y):
                        clear.alpha = 0
                        canvas = np.zeros((720,1280,3), np.uint8)
                        for cb in colors:
                            if cb.text != "Eraser":
                                color = cb.color
                                break
                    else:
                        clear.alpha = 0.5

            elif upFingers[0] and not upFingers[1]:
                if whiteBoard.isHover(x, y):
                    #drawing on the canvas
                    if px == 0 and py == 0:
                        cv2.circle(frame, positions[8], brushSize, color, brushSize)
                        px, py = positions[8]
                    if color == (0,0,0):
                        cv2.circle(frame, positions[8], eraserSize, color, -1)
                        cv2.circle(canvas, (px,py), eraserSize, color, eraserSize)
                    else:
                        cv2.circle(frame, positions[8], brushSize, color, brushSize)
                        # if predict text showing, don't allow draw
                        if Counter_TextPredict == 0:
                            cv2.line(canvas, (px,py), positions[8], color , brushSize)
                    px, py = positions[8]
            else:
                px, py = 0, 0


        # put the white board on the frame
        whiteBoard.drawRect(frame)
        ########### moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)

        if frames_pred == 0:
            contour_gs, _ = cv2.findContours(imgInv.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contour_gs) > 1:
                contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[1]
                # Check if the largest contour satisfy the condition of minimum area
                if  cv2.contourArea(contour) > 5000:
                    x, y, w, h = cv2.boundingRect(contour)
                    image = imgInv[y:y + h, x:x + w]
                    image = 255 - image
                    score, pred = inference(image, categories)
                    if (score > 0.8):
                        Counter_TextPredict = 50
                        canvas = np.zeros((720, 1280, 3), np.uint8)

        # Show prediction
        if Counter_TextPredict > 0:
            cv2.putText(frame, f"you are drawing ", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5,
                        cv2.LINE_AA)
            pred_path = os.path.join("images", pred + ".png")
            image_pred = cv2.imread(pred_path, -1)
            image_pred = cv2.resize(image_pred, (75 , 75))

            x_offset = 550
            y_offset = 10

            y1, y2 = y_offset, y_offset + image_pred.shape[0]
            x1, x2 = x_offset, x_offset + image_pred.shape[1]

            alpha_s = image_pred[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * image_pred[:, :, c] +
                                          alpha_l * frame[y1:y2, x1:x2, c])
            Counter_TextPredict -= 1



        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)
        cv2.imshow('video', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_app()