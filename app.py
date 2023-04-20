# Import kivy deps
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.popup import Popup
from datetime import datetime

# Warning system deps
from playsound import playsound

# CNN deps
import cv2
import tensorflow as tf
import numpy as np
# import segmentation_models as sm
# sm.set_framework('tf.keras')
# sm.framework()


# Define Widgets
class RoundedImage(Widget):
    pass


# Define Screens
class HomeWindow(Screen):
    Window.clearcolor = (208 / 255, 201 / 255, 192 / 255, 1)
    Window.size = (1280, 720)


class AboutWindow(Screen):
    pass


# get valid camera indices, allows us to switch between these indices for users with multiple
# camera devices attached to their machine
# credit: https://stackoverflow.com/a/61768256
def return_camera_indices():
    # checks the first 10 indexes.
    index = 0
    arr = []
    i = 10
    while i > 0:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
        i -= 1
    return arr


class SettingsPopup(Popup):
    def __init__(self, obj, **kwargs):
        super(SettingsPopup, self).__init__(**kwargs)
        self.obj = obj


class AppWindow(Screen):
    # App Window Widget References
    video_feed = ObjectProperty(None)
    activate_btn = ObjectProperty(None)
    activate_label = ObjectProperty(None)
    activate_btn_img = ObjectProperty(None)
    warning_label = ObjectProperty(None)

    def on_enter(self, *args):
        # set constants
        self.INFERENCE_RATE = 1.0 / 4.0  # trigger model inference 5 times per second
        self.CAMERA_RATE = 1.0 / 30.0  # update camera at 30.0 FPS
        self.warning_threshold = 0.3
        self.is_warning_active = True

        self.awarenessActivated = False
        self.capture = None
        self.model = None
        self.latest_mask = None

        self.warning_label.opacity = 0.0
        self.did_warn = False
        self.last_warn = datetime.timestamp(datetime.now())

        # check for camera devices
        self.cameraDevices = return_camera_indices()
        self.current_camera_index = 0

        print("Found {} cameras".format(len(self.cameraDevices)))

        if len(self.cameraDevices) <= 0:
            self.activate_btn.disabled = True
            self.activate_label.text = "No camera devices available, please connect a camera and try again"
            return

        # following MUST complete without error for system to work, otherwise, show error to user
        try:
            # Load CNN model
            self.model = tf.keras.models.load_model(
                './models/vgg_b_first.h5',
                # custom_objects={"iou_score": sm.metrics.IOUScore}
            )
            # Setup video capture device
            self.capture = cv2.VideoCapture(self.cameraDevices[0])
            Clock.schedule_interval(self.update_video_feed,
                                    self.CAMERA_RATE)
        except:
            self.capture = None
            self.activate_btn.disabled = True
            self.activate_label.text = "An exception occurred while loading model and/or loading capture device"
            print("An exception occurred while loading model and/or loading capture device")

    # Continuously update webcam window from camera feed
    def update_video_feed(self, *args):
        if self.capture is None:
            return

        # Get frame
        ret, frame = self.capture.read()

        if frame is None:
            return

        # Flip horizontal and convert image to texture
        frame_flipped = cv2.flip(frame, 0)
        frame_resized = cv2.resize(frame_flipped, (1280, 720)).tobytes()
        # frame_alphad = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2BGRA)
        # final_frame_buf = None
        # if self.latest_mask is not None and self.awarenessActivated is True:
        #     # something...
        #     final_frame_buf = cv2.addWeighted(frame_alphad, 1, self.latest_mask, 1, 0).tobytes()
        # else:
        #     final_frame_buf = frame_alphad.tobytes()

        img_texture = Texture.create(size=(1280, 720), colorfmt='bgr')
        img_texture.blit_buffer(frame_resized, colorfmt='bgr', bufferfmt='ubyte')
        self.video_feed.texture = img_texture

    def update_mask_feed(self, *args):
        if self.activate_label.text != 'Monitor is Active':
            return

        ret, image = self.capture.read()
        img_target = cv2.flip(image, 0)
        img_target_2 = cv2.resize(img_target,  (224, 224)) / 255.0
        img_target_3 = np.expand_dims(img_target_2, axis=0)

        pred = self.model.predict(img_target_3)[0]

        class_list = [
            'safe driving',
            'texting - right',
            'talking on the phone - right',
            'texting - left',
            'talking on the phone - left',
            'operating the radio',
            'drinking',
            'reaching behind',
            'hair and makeup',
            'talking to passenger'
        ]

        pred_class = class_list[pred.argmax(axis=-1)]

        if datetime.timestamp(datetime.now()) - self.last_warn > 10:
            self.last_warn = datetime.timestamp(datetime.now())
            print("warning trigger")

        print(pred_class)

        # try:
        #     #
        #     #
        #     #
        #     #
        #     # pred_class = class_list[pred.argmax(axis=-1)]
        #     #
        #     # print(pred_class)
        #
        #     # processing pred as mask
        #     # th, img_thresh = cv2.threshold(src=pred_img, thresh=0.5, maxval=255, type=cv2.THRESH_BINARY)
        #
        #     # new_img = np.expand_dims(img_thresh, axis=-1)
        #     #
        #     # # warning system, if proportion of mask is not big enough, WARN
        #     # if cv2.countNonZero(new_img) / 103680 < self.warning_threshold:
        #     #     self.warning_label.opacity = 1.0
        #     #
        #     #     # did warn ensures we only warn once per warn occurrence
        #     #     if self.did_warn == False:
        #     #         playsound('./assets/tesla_warn.mp3', False)
        #     #         self.did_warn = True
        #     # else:
        #     #     self.warning_label.opacity = 0.0
        #     #     self.did_warn = False
        #     #
        #     # new_img = new_img.astype(np.uint8)
        #     # final_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGRA)
        #     #
        #     # final_img = cv2.flip(final_img, 0)
        #     #
        #     # final_img = cv2.resize(final_img, (1280, 720))
        #     #
        #     # self.latest_mask = final_img
        # except:
        #     self.warning_label.opacity = 0.0
        #     self.activate_label.text = "Error running inference, will try again..."

    def update_settings(self, isWarningActive, warningThreshold, fpsValue):
        print("Update with new settings:")
        print("Warning Active: {}".format(str(isWarningActive)))
        print("Warning Threshold: {}".format(str(warningThreshold)))
        print("FPS Value: {}".format(str(fpsValue)))
        self.INFERENCE_RATE = 1.0 / float(fpsValue)
        self.is_warning_active = isWarningActive
        self.warning_threshold = float(warningThreshold)


    def open_settings(self):
        if self.awarenessActivated:
            self.toggle_awareness()
        settingspopup = SettingsPopup(self)
        settingspopup.open()

    def switch_camera(self):
        if len(self.cameraDevices) < 1: return
        if self.current_camera_index + 1 >= len(self.cameraDevices):
            self.current_camera_index = self.cameraDevices[0]
        else:
            self.current_camera_index += 1
            self.capture = cv2.VideoCapture(self.cameraDevices[self.current_camera_index])


    def toggle_awareness(self):
        # update toggles
        self.warning_label.opacity = 0.0
        self.awarenessActivated = not self.awarenessActivated
        if self.awarenessActivated:
            self.activate_btn.ids.activate_btn_img.source = './assets/stop_circle_FILL0_wght400_GRAD0_opsz48.png'
            self.activate_label.text = 'Monitor is Active'
            self.InferenceLoop = Clock.schedule_interval(self.update_mask_feed, self.INFERENCE_RATE)
        else:
            self.activate_btn.ids.activate_btn_img.source = './assets/play_circle_FILL0_wght400_GRAD0_opsz48.png'
            self.activate_label.text = 'Not Active'
            self.InferenceLoop.cancel()  # cancel the schedule_interval event


class WindowManager(ScreenManager):
    pass


class SafeTraxApp(App):
    pass


if __name__ == '__main__':
    SafeTraxApp().run()
