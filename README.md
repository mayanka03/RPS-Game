RPS (Rock Paper Scissors) with hand tracking

Run the asset generator once to create placeholder images: python generate_assets.py

Install dependencies (recommended in a virtualenv): pip install -r requirements.txt

Run the game: python Run.py

Notes:

The game uses your webcam and MediaPipe for hand detection.
Press 'q' to quit the game window.
If MediaPipe install fails on your system, let me know and I can provide an alternative lighter detector stub (less accurate).
