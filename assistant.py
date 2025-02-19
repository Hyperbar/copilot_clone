import base64
import cv2
import numpy as np  # Pour manipuler les tableaux d'images
import openai
import pyautogui
from threading import Lock, Thread
from cv2 import VideoCapture, imencode
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

load_dotenv()


class WebcamStream:
    def __init__(self):
        self.stream = VideoCapture(index=0)
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.stream.read()
            if not ret:
                continue
            with self.lock:
                self.frame = frame

    def read(self, encode=False):
        with self.lock:
            frame = self.frame.copy()
        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)
        return frame

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)
        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)
        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)
        with openai.audio.speech.with_streaming_response.create(
            model="tts-1", voice="alloy", response_format="pcm", input=response
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image
        provided by the user to answer its questions.
        Be friendly and helpful. Show some personality.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", [
                    {"type": "text", "text": "{prompt}"},
                    {"type": "image_url",
                     "image_url": "data:image/jpeg;base64,{image_base64}"},
                ]),
            ]
        )

        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain, lambda _: chat_message_history, input_messages_key="prompt", history_messages_key="chat_history"
        )


def take_screenshot():
    """
    Capture l'écran et renvoie l'image encodée en base64 (pour traitement)
    """
    screenshot = pyautogui.screenshot()
    # Convertir l'image en format BGR pour OpenCV
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    _, buffer = imencode(".jpeg", image)
    return base64.b64encode(buffer)


def take_screenshot_display():
    """
    Capture l'écran et renvoie l'image (pour affichage)
    """
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    return image


def get_image_source():
    choice = input(
        "Choisissez la source d'image (1: Webcam, 2: Capture d'écran): ")
    if choice == "1":
        return WebcamStream().start()  # Mode webcam
    elif choice == "2":
        return None  # Mode capture d'écran
    else:
        print("Choix invalide. Utilisation de la webcam par défaut.")
        return WebcamStream().start()


# Obtenir la source d'image choisie par l'utilisateur
image_source = get_image_source()

# Création du modèle et de l'assistant
model = ChatOpenAI(model="gpt-4o")
assistant = Assistant(model)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(
            audio, model="base", language="english")
        # Utiliser la bonne source d'image pour le traitement
        if image_source:
            image = image_source.read(encode=True)
        else:
            image = take_screenshot()
        assistant.answer(prompt, image)
    except UnknownValueError:
        print("Erreur de reconnaissance vocale.")


recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

if image_source:
    # Mode webcam : afficher un flux en direct dans une fenêtre unique
    window_title = "Vue en direct de la Webcam"
    # Créer une fenêtre redimensionnable
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    while True:
        frame = image_source.read()
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(1)
        # Quitte si 'q' ou Échap est appuyé
        if key == 27 or key == ord("q"):
            break
    image_source.stop()
    cv2.destroyAllWindows()
else:
    # Mode capture d'écran : afficher une capture d'une seule image statique
    window_title = "Capture d'écran"
    frame = take_screenshot_display()
    cv2.imshow(window_title, frame)
    cv2.waitKey(0)  # Attend une touche pour fermer
    cv2.destroyAllWindows()

stop_listening(wait_for_stop=False)
