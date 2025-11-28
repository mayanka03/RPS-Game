import cv2
import mediapipe as mp

class handDetector:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        # face detector (to avoid overlaying on the face)
        self.mpFace = mp.solutions.face_detection
        self.faceDetector = self.mpFace.FaceDetection(min_detection_confidence=0.6)

        # storage for last results
        self.results = None
        self.handType = []

    def findHands(self, img, draw=True):
        """Process the image for hands and faces. Stores results internally.
        Returns the annotated image. Optionally skips drawing hands that overlap with faces."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        # store handedness labels if available
        self.handType = []
        if self.results and self.results.multi_handedness:
            self.handType.extend([classification.classification[0].label for classification in self.results.multi_handedness])

        # detect faces
        self.faceResults = self.faceDetector.process(imgRGB)

        # draw hand landmarks only if they don't overlap with faces
        if self.results and self.results.multi_hand_landmarks:
            faceBoxes = self.findFaces(img)  # Get face bounding boxes
            for handLms in self.results.multi_hand_landmarks:
                handBox = self.getHandBoundingBox(img, handLms)  # Get hand bounding box
                if not self.isOverlapping(handBox, faceBoxes):  # Check for overlap
                    if draw:
                        self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=False):
        """Returns list of landmarks as [id, x, y] in pixel coords for the requested hand.
        If no hand detected, returns empty list."""
        lmList = []
        if hasattr(self, 'results') and self.results is not None and self.results.multi_hand_landmarks and handNo < len(self.results.multi_hand_landmarks):
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self, lmList, handNo=0):
        """Given a landmark list (as returned by findPosition), return a list of 5 booleans
        indicating which fingers are up: [thumb, index, middle, ring, pinky]
        This uses handedness information when available to interpret thumb direction."""
        if not lmList or len(lmList) < 21:
            return []

        fingers = []
        # Thumb: use handedness if available
        try:
            handLabel = self.handType[handNo].lower() if handNo < len(self.handType) else 'right'
        except Exception:
            handLabel = 'right'

        # Thumb: compare x coordinates of tip(4) and ip(3)
        if handLabel == 'right':
            fingers.append(lmList[4][1] < lmList[3][1])
        else:
            fingers.append(lmList[4][1] > lmList[3][1])

        # Other fingers: tip y < pip y means finger is up (y smaller => higher on image)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        fingers.extend([lmList[tip][2] < lmList[pip][2] for tip, pip in zip(tips, pips)])

        return fingers

    def findFaces(self, img):
        """Return list of face boxes in pixel coords: [x,y,w,h]"""
        boxes = []
        if not hasattr(self, 'faceResults') or self.faceResults is None:
            return boxes
        h, w, _ = img.shape
        if self.faceResults.detections:
            for det in self.faceResults.detections:
                bboxC = det.location_data.relative_bounding_box
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                bw = int(bboxC.width * w)
                bh = int(bboxC.height * h)
                boxes.append([x, y, bw, bh])
        return boxes

    def getHandBoundingBox(self, img, handLms):
        """Compute bounding box for a hand from its landmarks: [x, y, w, h]"""
        h, w, _ = img.shape
        x_coords = [int(lm.x * w) for lm in handLms.landmark]
        y_coords = [int(lm.y * h) for lm in handLms.landmark]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        return [x_min, y_min, x_max - x_min, y_max - y_min]  # [x, y, w, h]

    def isOverlapping(self, box1, boxes2):
        """Check if box1 [x,y,w,h] overlaps with any box in boxes2 (list of [x,y,w,h]).
        Uses axis-aligned bounding box (AABB) intersection."""
        x1, y1, w1, h1 = box1
        return any(
            x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2
            for x2, y2, w2, h2 in boxes2
        )
