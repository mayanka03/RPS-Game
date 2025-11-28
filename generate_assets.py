"""Generate placeholder images (Rock, Paper, Scissor) in the images/ folder.
This script requires OpenCV (cv2).
"""
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, 'images')
MUSIC_DIR = os.path.join(BASE_DIR, 'music')

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(MUSIC_DIR, exist_ok=True)

def make_image(name, color, text):
    path = os.path.join(IMG_DIR, name)
    img = np.full((480, 640, 3), color, dtype=np.uint8)
    cv2.putText(img, text, (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 8, cv2.LINE_AA)
    cv2.imwrite(path, img)
    print('Wrote', path)

if __name__ == '__main__':
    make_image('Rock.jpeg', (60,60,160), 'ROCK')
    make_image('Paper.jpeg', (60,160,60), 'PAPER')
    make_image('Scissor.jpeg', (160,60,60), 'SCISSOR')
    print('All images generated in', IMG_DIR)
