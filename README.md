# FACIAL-MOTION-CAPTURE-LIVE-ANIMATION
SENIOR DESIGN PROJECT

  Landmark_tracking.py file is for face detection and facial landmark tracking and it uses Opencv and dlib libraries. You can run it on Pycharm or any other python editor.

  On blender files, on blender, go scripting page and firstly you need to run OpenCVAnimOperator.py file, secondly you should run OpenCVAnim.py file and There will be a screen will be activated named OpenCV Animation. On that screen, press on Capture button. Output camera will be active and you can see your animation on the screen with do your some facial expressions like blink.

  The system starts with face detection. Face detection needs an input face and webcam. The second step is facial landmark tracking. Both of them use OpenCV and Haar cascade for face detection and tracking. These tracks also become facial motion capture data. 2D points of facial landmarks are data. All of them work on Blender by Blender scripts which are written in Python language. After facial motion capture data is taken by scripts, shape keys on the Python scripts become active and animation happens. OBS comes into the system at this stage, and it takes the animation to the virtual camera, and then it connects to the Zoom application. Finally, the Zoom application gives the system result to the screen. So the user's input face becomes an avatar face as an output.


