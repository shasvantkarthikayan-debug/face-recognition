from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.button import MDRaisedButton, MDIconButton, MDFillRoundFlatButton
from kivymd.uix.label import MDLabel
from kivymd.uix.textfield import MDTextField
from kivymd.uix.dialog import MDDialog
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.card import MDCard
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.widget import Widget
import cv2
import numpy as np
import mediapipe as mp
import os
import threading


class GradientWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self.update_gradient, size=self.update_gradient)
        self.update_gradient()
    
    def update_gradient(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            # Dark blue gradient
            Color(0.05, 0.15, 0.35, 1)
            Rectangle(pos=self.pos, size=self.size)
            
            Color(0.1, 0.2, 0.5, 0.6)
            Rectangle(pos=self.pos, size=(self.width, self.height * 0.7))
            
            Color(0.15, 0.25, 0.6, 0.3)
            Rectangle(pos=self.pos, size=(self.width, self.height * 0.4))


class HomeScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'home'
        
        # Gradient background
        gradient = GradientWidget()
        self.add_widget(gradient)
        
        layout = FloatLayout()
        
        # Title Card
        title_card = MDCard(
            orientation='vertical',
            size_hint=(0.9, 0.25),
            pos_hint={'center_x': 0.5, 'top': 0.95},
            elevation=10,
            md_bg_color=(0.1, 0.1, 0.2, 0.9)
        )
        
        title_box = MDBoxLayout(orientation='vertical', padding=20, spacing=10)
        
        title = MDLabel(
            text="🎯 FacePass",
            halign='center',
            font_style='H3',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1)
        )
        
        subtitle = MDLabel(
            text="Advanced Face Recognition System",
            halign='center',
            font_style='Subtitle1',
            theme_text_color='Custom',
            text_color=(0.7, 0.8, 1, 1)
        )
        
        title_box.add_widget(title)
        title_box.add_widget(subtitle)
        title_card.add_widget(title_box)
        
        # Buttons Card
        btn_card = MDCard(
            orientation='vertical',
            size_hint=(0.85, 0.5),
            pos_hint={'center_x': 0.5, 'center_y': 0.4},
            elevation=15,
            md_bg_color=(0.15, 0.15, 0.25, 0.95)
        )
        
        btn_layout = MDBoxLayout(orientation='vertical', padding=30, spacing=25)
        
        capture_btn = MDFillRoundFlatButton(
            text="  📸  Capture Faces",
            pos_hint={'center_x': 0.5},
            size_hint_x=0.9,
            font_size='18sp',
            md_bg_color=(0.2, 0.6, 0.9, 1),
            on_release=self.go_to_capture
        )
        
        train_btn = MDFillRoundFlatButton(
            text="  🧠  Train Model",
            pos_hint={'center_x': 0.5},
            size_hint_x=0.9,
            font_size='18sp',
            md_bg_color=(0.3, 0.7, 0.4, 1),
            on_release=self.go_to_train
        )
        
        recognize_btn = MDFillRoundFlatButton(
            text="  🔍  Start Recognition",
            pos_hint={'center_x': 0.5},
            size_hint_x=0.9,
            font_size='18sp',
            md_bg_color=(0.8, 0.3, 0.5, 1),
            on_release=self.go_to_recognize
        )
        
        btn_layout.add_widget(capture_btn)
        btn_layout.add_widget(train_btn)
        btn_layout.add_widget(recognize_btn)
        btn_card.add_widget(btn_layout)
        
        footer = MDLabel(
            text="Powered by MediaPipe & KivyMD",
            halign='center',
            font_style='Caption',
            size_hint_y=None,
            height=30,
            pos_hint={'center_x': 0.5, 'y': 0.02},
            theme_text_color='Custom',
            text_color=(0.5, 0.5, 0.6, 1)
        )
        
        layout.add_widget(title_card)
        layout.add_widget(btn_card)
        layout.add_widget(footer)
        
        self.add_widget(layout)
    
    def go_to_capture(self, instance):
        self.manager.current = 'capture'
    
    def go_to_train(self, instance):
        self.manager.current = 'train'
    
    def go_to_recognize(self, instance):
        self.manager.current = 'recognize'


class CaptureScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'capture'
        self.capture = None
        self.count = 0
        self.person_name = ""
        self.category = ""
        
        gradient = GradientWidget()
        self.add_widget(gradient)
        
        layout = FloatLayout()
        
        main_card = MDCard(
            orientation='vertical',
            size_hint=(0.95, 0.95),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            elevation=20,
            md_bg_color=(0.12, 0.12, 0.22, 0.98)
        )
        
        content = MDBoxLayout(orientation='vertical', padding=15, spacing=15)
        
        top_bar = MDBoxLayout(size_hint_y=0.08)
        back_btn = MDIconButton(
            icon='arrow-left',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1),
            on_release=self.go_back
        )
        title = MDLabel(
            text="📸 Capture Faces",
            halign='center',
            font_style='H5',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1)
        )
        top_bar.add_widget(back_btn)
        top_bar.add_widget(title)
        top_bar.add_widget(Widget())
        
        input_card = MDCard(
            size_hint_y=0.15,
            md_bg_color=(0.2, 0.2, 0.3, 0.7),
            elevation=5
        )
        
        input_box = MDBoxLayout(orientation='vertical', padding=10, spacing=10)
        
        self.name_field = MDTextField(
            hint_text="👤 Person's Name",
            mode="rectangle",
            size_hint_x=1,
            font_size='16sp'
        )
        
        self.category_field = MDTextField(
            hint_text="🏷️ Category (parents/students)",
            mode="rectangle",
            size_hint_x=1,
            font_size='16sp'
        )
        
        input_box.add_widget(self.name_field)
        input_box.add_widget(self.category_field)
        input_card.add_widget(input_box)
        
        camera_card = MDCard(
            size_hint_y=0.55,
            md_bg_color=(0.05, 0.05, 0.15, 1),
            elevation=10
        )
        camera_box = MDBoxLayout(padding=5)
        self.camera_image = Image()
        camera_box.add_widget(self.camera_image)
        camera_card.add_widget(camera_box)
        
        self.status_label = MDLabel(
            text="Enter name and category, then press Start 👇",
            halign='center',
            size_hint_y=0.06,
            theme_text_color='Custom',
            text_color=(0.8, 0.9, 1, 1),
            font_style='Body1'
        )
        
        btn_layout = MDBoxLayout(size_hint_y=0.12, spacing=10, padding=5)
        
        self.start_btn = MDFillRoundFlatButton(
            text="▶️ Start",
            md_bg_color=(0.2, 0.7, 0.3, 1),
            on_release=self.start_camera
        )
        
        self.capture_btn = MDFillRoundFlatButton(
            text="📷 Capture",
            disabled=True,
            md_bg_color=(0.3, 0.5, 0.9, 1),
            on_release=self.capture_photo
        )
        
        self.stop_btn = MDFillRoundFlatButton(
            text="⏹️ Stop",
            disabled=True,
            md_bg_color=(0.9, 0.3, 0.3, 1),
            on_release=self.stop_camera
        )
        
        btn_layout.add_widget(self.start_btn)
        btn_layout.add_widget(self.capture_btn)
        btn_layout.add_widget(self.stop_btn)
        
        content.add_widget(top_bar)
        content.add_widget(input_card)
        content.add_widget(camera_card)
        content.add_widget(self.status_label)
        content.add_widget(btn_layout)
        
        main_card.add_widget(content)
        layout.add_widget(main_card)
        self.add_widget(layout)
    
    def start_camera(self, instance):
        self.person_name = self.name_field.text.strip()
        self.category = self.category_field.text.strip().lower()
        
        if not self.person_name or self.category not in ['parents', 'students']:
            self.status_label.text = "❌ Invalid name or category!"
            return
        
        save_dir = f"known_faces/{self.category}/{self.person_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        self.capture = cv2.VideoCapture(0)
        self.start_btn.disabled = True
        self.capture_btn.disabled = False
        self.stop_btn.disabled = False
        self.count = 0
        
        Clock.schedule_interval(self.update_camera, 1.0 / 30.0)
        self.status_label.text = f"✅ Camera active. Press Capture to save photos."
    
    def update_camera(self, dt):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                buf = cv2.flip(frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.camera_image.texture = texture
    
    def capture_photo(self, instance):
        if self.capture and self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                save_dir = f"known_faces/{self.category}/{self.person_name}"
                path = f"{save_dir}/{self.person_name}_{self.count}.jpg"
                cv2.imwrite(path, frame)
                self.count += 1
                self.status_label.text = f"✅ Saved {self.count} photos successfully!"
    
    def stop_camera(self, instance):
        if self.capture:
            self.capture.release()
            Clock.unschedule(self.update_camera)
        
        self.start_btn.disabled = False
        self.capture_btn.disabled = True
        self.stop_btn.disabled = True
        self.status_label.text = f"🎉 Captured {self.count} photos total!"
    
    def go_back(self, instance):
        self.stop_camera(instance)
        self.manager.current = 'home'


class TrainScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'train'
        
        gradient = GradientWidget()
        self.add_widget(gradient)
        
        layout = FloatLayout()
        
        main_card = MDCard(
            orientation='vertical',
            size_hint=(0.9, 0.7),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            elevation=20,
            md_bg_color=(0.12, 0.12, 0.22, 0.98)
        )
        
        content = MDBoxLayout(orientation='vertical', padding=20, spacing=20)
        
        top_bar = MDBoxLayout(size_hint_y=0.15)
        back_btn = MDIconButton(
            icon='arrow-left',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1),
            on_release=self.go_back
        )
        title = MDLabel(
            text="🧠 Train Model",
            halign='center',
            font_style='H5',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1)
        )
        top_bar.add_widget(back_btn)
        top_bar.add_widget(title)
        top_bar.add_widget(Widget())
        
        status_card = MDCard(
            size_hint_y=0.6,
            md_bg_color=(0.2, 0.2, 0.3, 0.8),
            elevation=5
        )
        
        status_box = MDBoxLayout(padding=20)
        self.status_label = MDLabel(
            text="Press the button below to train\n\n🎯 Process all captured faces\n⚡ May take a few moments",
            halign='center',
            theme_text_color='Custom',
            text_color=(0.9, 0.9, 1, 1),
            font_style='Body1'
        )
        status_box.add_widget(self.status_label)
        status_card.add_widget(status_box)
        
        train_btn = MDFillRoundFlatButton(
            text="🚀 Start Training",
            pos_hint={'center_x': 0.5},
            size_hint=(0.8, 0.15),
            font_size='18sp',
            md_bg_color=(0.3, 0.7, 0.4, 1),
            on_release=self.start_training
        )
        
        content.add_widget(top_bar)
        content.add_widget(status_card)
        content.add_widget(train_btn)
        
        main_card.add_widget(content)
        layout.add_widget(main_card)
        self.add_widget(layout)
        
        self.mp_face_mesh = mp.solutions.face_mesh
    
    def get_face_embedding(self, image):
        with self.mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            embedding = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
            return embedding
    
    def train_model(self):
        Clock.schedule_once(lambda dt: setattr(
            self.status_label, 'text', 
            "🔄 Training in progress...\n\nPlease wait..."
        ))
        
        averaged_encodings = []
        names_list = []
        categories_list = []
        
        base_dir = "known_faces"
        processed = 0
        
        for category in os.listdir(base_dir):
            cat_path = os.path.join(base_dir, category)
            if not os.path.isdir(cat_path):
                continue
            
            for person in os.listdir(cat_path):
                person_path = os.path.join(cat_path, person)
                if not os.path.isdir(person_path):
                    continue
                
                person_embeddings = []
                for img_file in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    embedding = self.get_face_embedding(img)
                    if embedding is not None:
                        person_embeddings.append(embedding)
                        processed += 1
                        Clock.schedule_once(lambda dt, p=processed: setattr(
                            self.status_label, 'text', 
                            f"🔄 Processing...\n\n📊 {p} images analyzed"
                        ))
                
                if person_embeddings:
                    avg_embedding = np.mean(person_embeddings, axis=0)
                    averaged_encodings.append(avg_embedding)
                    names_list.append(person)
                    categories_list.append(category)
        
        np.save("encodings.npy", averaged_encodings)
        np.save("names.npy", names_list)
        np.save("categories.npy", categories_list)
        
        Clock.schedule_once(lambda dt: setattr(
            self.status_label, 'text',
            f"✅ Training Complete!\n\n📊 {processed} images\n👥 {len(names_list)} people\n\n🎉 Ready!"
        ))
    
    def start_training(self, instance):
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def go_back(self, instance):
        self.manager.current = 'home'


class RecognizeScreen(MDScreen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'recognize'
        self.capture = None
        self.is_running = False
        
        try:
            self.known_encodings = np.load("encodings.npy", allow_pickle=True)
            self.known_names = np.load("names.npy", allow_pickle=True)
            self.known_categories = np.load("categories.npy", allow_pickle=True)
        except:
            self.known_encodings = []
            self.known_names = []
            self.known_categories = []
        
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1
        )
        
        self.correct = 0
        self.total = 0
        
        gradient = GradientWidget()
        self.add_widget(gradient)
        
        layout = FloatLayout()
        
        main_card = MDCard(
            orientation='vertical',
            size_hint=(0.95, 0.95),
            pos_hint={'center_x': 0.5, 'center_y': 0.5},
            elevation=20,
            md_bg_color=(0.12, 0.12, 0.22, 0.98)
        )
        
        content = MDBoxLayout(orientation='vertical', padding=15, spacing=15)
        
        top_bar = MDBoxLayout(size_hint_y=0.08)
        back_btn = MDIconButton(
            icon='arrow-left',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1),
            on_release=self.go_back
        )
        title = MDLabel(
            text="🔍 Face Recognition",
            halign='center',
            font_style='H5',
            theme_text_color='Custom',
            text_color=(1, 1, 1, 1)
        )
        top_bar.add_widget(back_btn)
        top_bar.add_widget(title)
        top_bar.add_widget(Widget())
        
        camera_card = MDCard(
            size_hint_y=0.7,
            md_bg_color=(0.05, 0.05, 0.15, 1),
            elevation=10
        )
        camera_box = MDBoxLayout(padding=5)
        self.camera_image = Image()
        camera_box.add_widget(self.camera_image)
        camera_card.add_widget(camera_box)
        
        info_card = MDCard(
            size_hint_y=0.08,
            md_bg_color=(0.2, 0.2, 0.3, 0.8),
            elevation=5
        )
        info_box = MDBoxLayout(padding=10)
        self.info_label = MDLabel(
            text="Press Start to begin recognition 🚀",
            halign='center',
            theme_text_color='Custom',
            text_color=(0.9, 0.9, 1, 1),
            font_style='Body1'
        )
        info_box.add_widget(self.info_label)
        info_card.add_widget(info_box)
        
        btn_layout = MDBoxLayout(size_hint_y=0.1, spacing=10, padding=5)
        
        self.start_btn = MDFillRoundFlatButton(
            text="▶️ Start",
            md_bg_color=(0.2, 0.7, 0.3, 1),
            on_release=self.start_recognition
        )
        
        self.search_btn = MDFillRoundFlatButton(
            text="🔎 Search",
            disabled=True,
            md_bg_color=(0.6, 0.4, 0.9, 1),
            on_release=self.search_unknown
        )
        
        self.stop_btn = MDFillRoundFlatButton(
            text="⏹️ Stop",
            disabled=True,
            md_bg_color=(0.9, 0.3, 0.3, 1),
            on_release=self.stop_recognition
        )
        
        btn_layout.add_widget(self.start_btn)
        btn_layout.add_widget(self.search_btn)
        btn_layout.add_widget(self.stop_btn)
        
        content.add_widget(top_bar)
        content.add_widget(camera_card)
        content.add_widget(info_card)
        content.add_widget(btn_layout)
        
        main_card.add_widget(content)
        layout.add_widget(main_card)
        self.add_widget(layout)
        
        self.current_frame = None
        self.current_name = "Unknown"
        self.face_bbox = None
    
    def l2_distance(self, a, b):
        return np.linalg.norm(a - b)
    
    def get_face_embedding(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        embedding = np.array([[p.x, p.y, p.z] for p in landmarks]).flatten()
        return embedding
    
    def start_recognition(self, instance):
        if not len(self.known_encodings):
            self.info_label.text = "❌ No training data! Train first."
            return
        
        self.capture = cv2.VideoCapture(0)
        self.is_running = True
        self.start_btn.disabled = True
        self.search_btn.disabled = False
        self.stop_btn.disabled = False
        self.correct = 0
        self.total = 0
        
        Clock.schedule_interval(self.update_recognition, 1.0 / 30.0)
    
    def update_recognition(self, dt):
        if not self.is_running or not self.capture.isOpened():
            return
        
        ret, frame = self.capture.read()
        if not ret:
            return
        
        self.current_frame = frame.copy()
        embedding = self.get_face_embedding(frame)
        
        if embedding is not None:
            distances = [self.l2_distance(embedding, enc) for enc in self.known_encodings]
            min_idx = np.argmin(distances)
            
            h, w, _ = frame.shape
            xs = [p[0] for p in embedding.reshape(-1, 3)]
            ys = [p[1] for p in embedding.reshape(-1, 3)]
            left, right = int(min(xs) * w), int(max(xs) * w)
            top, bottom = int(min(ys) * h), int(max(ys) * h)
            
            if distances[min_idx] < 5:
                name = self.known_names[min_idx]
                category = self.known_categories[min_idx]
                self.correct += 1
                box_color = (0, 255, 0)
                self.current_name = name
            else:
                name = "Unknown"
                category = ""
                box_color = (0, 0, 255)
                self.current_name = "Unknown"
            
            self.face_bbox = (left, top, right, bottom)
            self.total += 1
            accuracy = (self.correct / self.total) * 100
            
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, 3)
            cv2.putText(frame, f"{name} ({category})",
                       (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
            self.info_label.text = f"👤 {name} ({category}) | 📊 {accuracy:.1f}%"
        
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.camera_image.texture = texture
    
    def search_unknown(self, instance):
        if self.current_name != "Unknown" or self.current_frame is None:
            return
        
        self.info_label.text = "🔍 Searching..."
        threading.Thread(target=self.perform_search, daemon=True).start()
    
    def perform_search(self):
        if self.face_bbox:
            left, top, right, bottom = self.face_bbox
            face_crop = self.current_frame[top:bottom, left:right]
            filename = f"unknown_{np.random.randint(10000)}.jpg"
            cv2.imwrite(filename, face_crop)
            
            Clock.schedule_once(lambda dt: setattr(
                self.info_label, 'text',
                f"✅ Saved as {filename}"
            ))
    
    def stop_recognition(self, instance):
        self.is_running = False
        if self.capture:
            self.capture.release()
            Clock.unschedule(self.update_recognition)
        
        self.start_btn.disabled = False
        self.search_btn.disabled = True
        self.stop_btn.disabled = True
        self.info_label.text = "⏹️ Stopped"
    
    def go_back(self, instance):
        self.stop_recognition(instance)
        self.manager.current = 'home'


class FaceRecognitionApp(MDApp):
    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        self.title = "FacePass"
        
        sm = MDScreenManager()
        sm.add_widget(HomeScreen())
        sm.add_widget(CaptureScreen())
        sm.add_widget(TrainScreen())
        sm.add_widget(RecognizeScreen())
        
        return sm


if __name__ == '__main__':
    FaceRecognitionApp().run()