import bpy
import cv2
import time
import numpy


# Download trained model (lbfmodel.yaml)
# https://github.com/kurnianggoro/GSOC2017/tree/master/data


class OpenCVAnimOperator(bpy.types.Operator):
    """Operator which runs its self from a timer"""
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"

    # Set paths to trained models downloaded above
    face_detect_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    landmark_model_path = "D:\\Downloads\\lbfmodel.yaml"  # This file need to be located on the same folder

    # Load models
    fm = cv2.face.createFacemarkLBF()
    fm.loadModel(landmark_model_path)
    cas = cv2.CascadeClassifier(face_detect_path)

    _timer = None
    _cap = None
    stop = False

    # Webcam resolution:
    width = 640
    height = 480

    # 3D model points.
    model_points = numpy.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=numpy.float32)
    # Camera internals
    camera_matrix = numpy.array(
        [[height, 0.0, width / 2],
         [0.0, height, height / 2],
         [0.0, 0.0, 1.0]], dtype=numpy.float32
    )

    # The main "loop"
    def modal(self, context, event):

        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}

        if event.type == 'TIMER':
            self.init_camera()
            _, image = self._cap.read()
            # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)

            # find faces
            faces = self.cas.detectMultiScale(image,
                                              scaleFactor=1.05,
                                              minNeighbors=3,
                                              flags=cv2.CASCADE_SCALE_IMAGE,
                                              minSize=(int(self.width / 5), int(self.width / 5)))

            # find biggest face, and only keep it
            if type(faces) is numpy.ndarray and faces.size > 0:
                biggestFace = numpy.zeros(shape=(1, 4))
                for face in faces:
                    if face[2] > biggestFace[0][2]:
                        print(face)
                        biggestFace[0] = face

                # find the landmarks.
                _, landmarks = self.fm.fit(image, faces=biggestFace)

                for mark in landmarks:
                    shape = mark[0]

                    # 2D image points. If you change the image, you need to change vector
                    image_points = numpy.array([shape[30],  # Nose tip - 31
                                                shape[8],  # Chin - 9
                                                shape[36],  # Left eye left corner - 37
                                                shape[45],  # Right eye right corne - 46
                                                shape[48],  # Left Mouth corner - 49
                                                shape[54]  # Right mouth corner - 55
                                                ], dtype=numpy.float32)

                    dist_coeffs = numpy.zeros((4, 1))  # Assuming no lens distortion

                    difference_eye1 = numpy.linalg.norm(
                        shape[43] - shape[47])  # euclidean distance between right eye eyelid points
                    difference_eye2 = numpy.linalg.norm(
                        shape[44] - shape[46])  # euclidean distance between right eye eyelid points

                    difference_eye3 = numpy.linalg.norm(shape[42] - shape[
                        45])  # # euclidean distance between right eye eyelid points, shape[45] = right corner of right eye.

                    EAR_ratio = (difference_eye1 + difference_eye2) / (
                                2 * difference_eye3)  # right eye aspect ratio formula

                    difference_eye4 = numpy.linalg.norm(
                        shape[37] - shape[41])  # euclidean distance between left eye eyelid points
                    difference_eye5 = numpy.linalg.norm(
                        shape[38] - shape[40])  # euclidean distance between left eye eyelid points

                    difference_eye6 = numpy.linalg.norm(shape[36] - shape[
                        39])  # # euclidean distance between left eye eyelid points, shape[36] = left corner of left eye.

                    EAR_ratio2 = (difference_eye4 + difference_eye5) / (
                                2 * difference_eye6)  # right eye aspect ratio formula

                    EAR_total_ratio = (EAR_ratio + EAR_ratio2) / 2

                    # print(EAR_total_ratio)

                    # bpy.data.shape_keys["Key"].key_blocks["mouth_open"].value = 1 #mouth_open_shape_key

                    if EAR_total_ratio < 0.25:  # if EAR_ratio is close to zero, that means eye blink happens
                        bpy.data.shape_keys["Key"].key_blocks[
                            "Blink"].value = 1  # eye_blink_shape_key is activated (eye blink happens)
                    else:
                        bpy.data.shape_keys["Key"].key_blocks[
                            "Blink"].value = 0  # eye_blink_shape_key is not activated (eyes are open)

                    # bpy.data.shape_keys["Key"].key_blocks["eye_brows_up"].value = 1 #eye_brows_up_shape_key

                    difference_mouth1 = numpy.linalg.norm(
                        shape[61] - shape[67])  # euclidean distance between mouth inner points
                    difference_mouth2 = numpy.linalg.norm(
                        shape[62] - shape[66])  # euclidean distance between mouth inner points
                    difference_mouth3 = numpy.linalg.norm(
                        shape[63] - shape[65])  # euclidean distance between mouth inner points

                    difference_mouth4 = numpy.linalg.norm(shape[60] - shape[
                        64])  # euclidean distance between mouth inner points, shape[64] = right corner of inner mouth.

                    MIR_ratio = (difference_mouth1 + difference_mouth2 + difference_mouth3) / (
                                3 * difference_mouth4)  # Mouth inner aspect formula added by me, since the two movements are similar to each other, it is applied by analogy with the eye aspect ratio formula.

                    difference_mouth_out1 = numpy.linalg.norm(
                        shape[49] - shape[59])  # euclidean distance between mouth outer points
                    difference_mouth_out2 = numpy.linalg.norm(
                        shape[50] - shape[58])  # euclidean distance between mouth outer points
                    difference_mouth_out3 = numpy.linalg.norm(
                        shape[51] - shape[57])  # euclidean distance between mouth outer points
                    difference_mouth_out4 = numpy.linalg.norm(
                        shape[52] - shape[56])  # euclidean distance between mouth outer points
                    difference_mouth_out5 = numpy.linalg.norm(
                        shape[53] - shape[55])  # euclidean distance between mouth outer points

                    difference_mouth_out6 = numpy.linalg.norm(
                        shape[48] - shape[54])  # euclidean distance between mouth outer points

                    MOR_ratio = (difference_mouth_out1 + difference_mouth_out2 + difference_mouth_out3
                                 + difference_mouth_out4 + difference_mouth_out5) / (
                                            5 * difference_mouth_out6)  # Mouth outer aspect formula added by me, since the two movements are similar to each other, it is applied by analogy with the eye aspect ratio formula.

                    Mouth_total_ratio = (
                                                    MIR_ratio + MOR_ratio) / 2  # the average of inner and outer mouth aspect ratios.

                    # print (Mouth_total_ratio)

                    if Mouth_total_ratio < 0.25:  # if Mouth_total_ratio is close to zero, that means mouth is closed.
                        bpy.data.shape_keys["Key"].key_blocks[
                            "JawOpen"].value = 0  # mouth_open_shape_key is not activated so mouth is closed (mouth is closed)
                    else:
                        bpy.data.shape_keys["Key"].key_blocks[
                            "JawOpen"].value = 1  # mouth_open_shape_key is activated (open mouth)

                    # draw face markers
                    for (x, y) in shape:
                        cv2.circle(image, (x, y), 2, (0, 255, 255), -1)

            # draw detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            # Show camera image in a window
            cv2.imshow("Output", image)
            cv2.waitKey(1)

        return {'PASS_THROUGH'}

    def init_camera(self):
        if self._cap == None:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)

    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)

    def execute(self, context):
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None


def register():
    bpy.utils.register_class(OpenCVAnimOperator)


def unregister():
    bpy.utils.unregister_class(OpenCVAnimOperator)


if __name__ == "__main__":
    register()

    # test call
    # bpy.ops.wm.opencv_operator()


