import cv2 # pyright: ignore[reportMissingImports]
import os
import time
import pygame # pyright: ignore[reportMissingImports]
import hand_detector as hand_detector
import RPSGame
import sys
import numpy as np # pyright: ignore[reportMissingImports]
from generate_assets import make_image
from contextlib import suppress

def Run():
    pygame.init()
    # Use script-relative paths for assets
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    def img_path(name):
        return os.path.join(BASE_DIR, 'images', name)
    def music_path(name):
        return os.path.join(BASE_DIR, 'music', name)
    # Optional --wait flag to allow user to bring windows forward
    if '--wait' in sys.argv:
        with suppress(Exception):
            input('[RPS] --wait given: press Enter when ready to start (bring camera/window forward)')

    # Try loading music if available
    try:
        mpath = music_path('foo.wav')
        if os.path.exists(mpath):
            pygame.mixer.music.load(mpath)
            pygame.mixer.music.play(-1)
        else:
            print('[RPS] Warning: music/foo.wav not found, continuing without background music')
    except Exception as e:
        print('[RPS] Warning: failed to load/play music:', e)

    pTime = 0
    detector = hand_detector.handDetector(detectionCon=0.75)
    wCam, hCam = 640,480
    cap = cv2.VideoCapture(0)
    cap.set(3,wCam)
    cap.set(4,hCam)
    print('[RPS] Camera open status:', cap.isOpened())
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    count = 0

    # load images using script-relative paths
    rock = cv2.imread(img_path('Rock.jpeg'))
    paper = cv2.imread(img_path('Paper.jpeg'))
    scissor = cv2.imread(img_path('Scissor.jpeg'))

    # If any image fails to load, create a placeholder BEFORE scaling
    def placeholder(text, w=640, h=480):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, text, (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 8)
        return img

    if rock is None:
        print('[RPS] Warning: Rock image not found, creating placeholder')
        rock = placeholder('ROCK')
    if paper is None:
        print('[RPS] Warning: Paper image not found, creating placeholder')
        paper = placeholder('PAPER')
    if scissor is None:
        print('[RPS] Warning: Scissor image not found, creating placeholder')
        scissor = placeholder('SCISSOR')

    # scale overlays to be smaller so they don't block the webcam view
    def scale_overlay(img, max_w=240, max_h=180):
        if img is None:
            return None
        h, w = img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    rock_s = scale_overlay(rock)
    paper_s = scale_overlay(paper)
    scissor_s = scale_overlay(scissor)

    # ensure overlays are valid (fallback to small placeholder)
    if rock_s is None:
        rock_s = scale_overlay(placeholder('ROCK', 240, 180))
    if paper_s is None:
        paper_s = scale_overlay(placeholder('PAPER', 240, 180))
    if scissor_s is None:
        scissor_s = scale_overlay(placeholder('SCISSOR', 240, 180))

    overlaylist=[scissor_s,paper_s,rock_s]
    i=0
    computer_score=0
    player_score=0

    while True:
        success,img = cap.read()
        if not success or img is None:
            print('[RPS] Warning: failed to read frame from camera')
            time.sleep(0.1)
            continue
        # mirror the image so it feels like a mirror to the user
        img = cv2.flip(img, 1)

        img = detector.findHands(img)
        myList = detector.findPosition(img,draw=False)

        # draw face boxes so user knows where their face is
        faces = detector.findFaces(img)
        for (fx,fy,fw,fh) in faces:
            cv2.rectangle(img, (fx,fy), (fx+fw, fy+fh), (0,255,255), 2)

        # compute current fingers state for realtime debug display
        fingers_now = detector.fingersUp(myList, handNo=0) if myList else []
        # draw small indicators for thumb,index,middle,ring,pinky
        if fingers_now:
            labels = ['T','I','M','R','P']
            base_x = 60
            base_y = hCam - 40
            for idx, val in enumerate(fingers_now):
                cx = base_x + idx * 40
                color = (0,200,0) if val else (0,0,200)
                cv2.circle(img, (cx, base_y), 12, color, -1)
                cv2.putText(img, labels[idx], (cx-10, base_y+30), cv2.FONT_HERSHEY_PLAIN, 1.2, (255,255,255), 1)

        # decide overlay placement so it doesn't cover the face
        hO, wO, cO = overlaylist[i].shape
        # candidate positions: top-left, top-right, bottom-left, bottom-right
        candidates = [ (0,0), (wCam-wO,0), (0,hCam-hO), (wCam-wO,hCam-hO) ]

        def intersects(a, b):
            ax,ay,aw,ah = a
            bx,by,bw,bh = b
            return ax+aw > bx and bx+bw > ax and ay+ah > by and by+bh > ay

        chosen = candidates[0]
        overlayRect = (chosen[0], chosen[1], wO, hO)
        # pick a candidate that doesn't intersect any face; if none, keep top-left
        for cand in candidates:
            rect = (cand[0], cand[1], wO, hO)
            bad = False
            for f in faces:
                if intersects(rect, f):
                    bad = True
                    break
            if not bad:
                chosen = cand
                break

        # place overlay
        ox, oy = chosen
        img[oy:oy+hO, ox:ox+wO] = overlaylist[i]
        status = ""
        count+=1
        if(count<=20):
            cv2.putText(img,"1",(250,250),cv2.FONT_HERSHEY_COMPLEX,2.5,(36,255,12),6)
        elif(count<=40):
            cv2.putText(img,"2",(250,250),cv2.FONT_HERSHEY_COMPLEX,2.5,(36,255,12),6)
        elif(count<60):
            cv2.putText(img,"3",(250,250),cv2.FONT_HERSHEY_COMPLEX,2.5,(36,255,12),6)
        if(count==60):
            if len(myList) !=0 :
                try:
                    fingers = detector.fingersUp(myList, handNo=0)
                    cnt = sum(fingers)

                    # default to rock
                    player_choice = 1
                    det_text = 'Rock'
                    if not fingers:
                        player_choice = 1
                        det_text = 'Rock'
                    else:
                        # all five fingers up -> paper
                        if all(fingers): 
                            player_choice = 2
                            det_text = 'Paper'
                        # index + middle -> scissor
                        elif fingers[1] and fingers[2] and (not fingers[3]) and (not fingers[4]):
                            player_choice = 3
                            det_text = 'Scissor'
                        # no fingers -> rock
                        elif cnt == 0: # pyright: ignore[reportUndefinedVariable]
                            player_choice = 1
                            det_text = 'Rock'
                        else:
                            # fallback heuristics
                            if cnt >= 3: # pyright: ignore[reportUndefinedVariable]
                                player_choice = 2
                                det_text = 'Paper'
                            else:
                                player_choice = 1
                                det_text = 'Rock'

                    cv2.putText(img, det_text, (200,200), cv2.FONT_HERSHEY_COMPLEX, 3, (36,255,12), 4)
                    status,player_score,computer_score,computer=RPSGame.Game(player_choice,player_score,computer_score)
                    # set overlay index according to computer choice
                    if computer==1:
                        i=2   
                    elif computer==2:
                        i=1
                    else:
                        i=0             
                except Exception as e:
                    print('[RPS] Warning: detection/indexing error', e)
        elif(count<70):
            cv2.putText(img,status,(250,250),cv2.FONT_HERSHEY_COMPLEX,2.5,(36,255,12),6)                  
        elif(count==70):
            count=0           
        st=f"Computer Score : {computer_score}"
        cv2.putText(img,st,(30,440),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
        st=f"Player Score {player_score}"
        cv2.putText(img,st,(430,440),cv2.FONT_HERSHEY_PLAIN,1.5,(255,0,0),2)
        cTime = time.time() 
        fps = 1/(cTime-pTime) if (cTime-pTime)>0 else 0
        pTime=cTime
        cv2.putText(img,f'fps : {int(fps)}',(300,70),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
        cv2.imshow("Image",img)
        ch = cv2.waitKey(1) & 0xFF
        if ch == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Run()