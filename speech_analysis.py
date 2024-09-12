import speech_recognition as sr
from transformers import pipeline
import logging

class SpeechAnalyzer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    
    def capture_and_analyze_speech(self, stop_event):
        """Capture and analyze speech continuously."""
        try:
            with self.mic as source:
                print("Adjusting for ambient noise... Please wait.")
                self.recognizer.adjust_for_ambient_noise(source)
                print("Listening... Press Ctrl+C to stop.")
                while not stop_event.is_set():
                    try:
                        audio = self.recognizer.listen(source)
                        text = self.recognizer.recognize_google(audio)
                        print(f"Speech recognized: {text}")
                        sentiment = self.sentiment_analyzer(text)
                        print(f"Sentiment: {sentiment[0]['label']} with confidence score {sentiment[0]['score']}")
                    except sr.UnknownValueError:
                        logging.error("Google Web Speech API could not understand audio.")
                    except sr.RequestError as e:
                        logging.error(f"Could not request results from Google Web Speech API; {e}")
                    except Exception as e:
                        logging.error(f"Error during speech analysis: {e}")
        except KeyboardInterrupt:
            stop_event.set()
