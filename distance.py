import base64

from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2


class Distance:

    def __init__(self):
        pass

    @staticmethod
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    @staticmethod
    def get_distance_image(encoded_data):
        # image = cv2.imread("teste_4.jpeg")
        # cropped_image = image[170:450, :800]

        try:
            encoded_data = encoded_data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            image = cv2.imdecode(nparr, cv2.COLOR_RGB2BGR)
            cropped_image = image[20:150, :500]

            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)

            edged = cv2.Canny(gray, 100, 350)
            edged = cv2.dilate(edged, None, iterations=1)
            edged = cv2.erode(edged, None, iterations=1)

            # cv2.imshow("Image", edged)
            # cv2.waitKey(0)

            cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)

            (cnts, _) = contours.sort_contours(cnts)
            colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
                      (255, 0, 255))
            refObj = None

            min_dist_obstacle = 1000000
            height_obstacle = 1000000
            width_obstacle = 1000000
            for c in cnts:
                if cv2.contourArea(c) < 200:
                    continue
                box = cv2.minAreaRect(c)
                box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                cX = np.average(box[:, 0])
                cY = np.average(box[:, 1])
                if refObj is None:
                    (tl, tr, br, bl) = box
                    (tlblX, tlblY) = Distance.midpoint(tl, bl)
                    (trbrX, trbrY) = Distance.midpoint(tr, br)
                    D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
                    refObj = (box, (cX, cY), D / 0.8)
                    continue

                orig = cropped_image.copy()
                cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
                cv2.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)

                D_middle = dist.euclidean(refObj[1], (cX, cY)) / refObj[2]
                if D_middle < min_dist_obstacle:
                    min_dist_obstacle = D_middle
                    height_obstacle = (box[len(box) - 1][1] - box[0][1]) / refObj[2]
                    width_obstacle = (box[1][0] - box[0][0]) / refObj[2]

            # print(min_dist_obstacle, height_obstacle, width_obstacle)
            return min_dist_obstacle, height_obstacle, width_obstacle
        except (Exception, ) as e:
            pass

