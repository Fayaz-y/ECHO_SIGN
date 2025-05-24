import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import time
import threading
from spell import l  # Importing the set of valid words

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Video Player")
        self.root.geometry("800x600")
        
        self.video_folder = "assets1"  # Folder containing sign language videos
        self.video_queue = []  # List to store video filenames
        
        tk.Label(root, text="Enter Sentence:", font=("Arial", 14)).pack(pady=10)
        
        self.text_input = tk.Entry(root, font=("Arial", 14), width=50)
        self.text_input.pack(pady=10)
        
        self.process_button = tk.Button(root, text="Process & Play", font=("Arial", 12), command=self.process_sentence)
        self.process_button.pack(pady=10)
        
        self.video_label = tk.Label(root, text="", font=("Arial", 14))
        self.video_label.pack(pady=10)
        
        self.exit_button = tk.Button(root, text="Exit", font=("Arial", 12), command=root.quit)
        self.exit_button.pack(pady=10)
    
    def process_sentence(self):
        sentence = self.text_input.get().strip()
        if not sentence:
            messagebox.showerror("Error", "Please enter a sentence!")
            return
        
        words = sentence.split(" ")  # Split sentence into words
        wordss = []

        for i in words:
            if i not in l:
                for j in i:
                    wordss.append(j)
            else:
                wordss.append(i)
        self.video_queue.clear()
        
        for word in wordss:
            if word.lower() in l:  # Check if the word is valid using the set 'l'
                if self.video_exists(word):
                    self.video_queue.append(word)
                else:  # If no video exists for the valid word, split into letters
                    self.add_letters_to_queue(word)
            else:  # If the word is invalid, split into letters
                self.add_letters_to_queue(word)
        
        if self.video_queue:
            threading.Thread(target=self.play_videos, daemon=True).start()
    
    def add_letters_to_queue(self, word):
        for letter in word:
            if self.video_exists(letter):
                self.video_queue.append(letter)
    
    def video_exists(self, name):
        return os.path.exists(os.path.join(self.video_folder, f"{name}.mp4"))
    
    def play_videos(self):
        for video_name in self.video_queue:
            video_path = os.path.join(self.video_folder, f"{video_name}.mp4")
            self.video_label.config(text=f"Playing: {video_name}")
            
            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (600, 400))
                cv2.imshow("Sign Language Video", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            cap.release()
            time.sleep(0.5)
        
        cv2.destroyAllWindows()
        self.video_label.config(text="Playback Complete")

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()