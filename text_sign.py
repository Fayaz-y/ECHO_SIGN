import os
import time
import threading
import cv2
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Button, PhotoImage, messagebox, Label
from spell import l

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"D:\KPR_Hackathon\ECHO_SIGN\assets_text\frame0")

def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.geometry("800x600")
        self.window.configure(bg="#FFFFFF")
        self.window.resizable(False, False)

        self.canvas = Canvas(
            self.window,
            bg="#FFFFFF",
            height=600,
            width=800,
            bd=0,
            highlightthickness=0,
            relief="ridge"
        )
        self.canvas.place(x=0, y=0)

        # Load images
        self.image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
        self.canvas.create_image(400.0, 50.0, image=self.image_image_1)

        self.image_image_2 = PhotoImage(file=relative_to_assets("image_2.png"))
        self.canvas.create_image(228.0, 153.0, image=self.image_image_2)

        # Entry field with enhanced styling
        self.entry_image_1 = PhotoImage(file=relative_to_assets("entry_1.png"))
        self.canvas.create_image(400.0, 220.5, image=self.entry_image_1)
        self.entry = Entry(
            bd=2,  # Add a border
            bg="#F0F0F0",  # Light gray background
            fg="#888888",  # Placeholder text color
            highlightthickness=1,
            highlightcolor="#4A90E2",  # Blue highlight when focused
            font=("Arial", 14),  # Larger, more readable font
            justify='center'  # Center-align text
        )
        self.entry.place(
            x=122.0,
            y=189.0,
            width=556.0,
            height=61.0
        )
        
        # Add placeholder text
        self.entry.insert(0, "Enter sentence here...")
        self.entry.bind('<FocusIn>', self.on_entry_click)
        self.entry.bind('<FocusOut>', self.on_focusout)

        self.image_image_3 = PhotoImage(file=relative_to_assets("image_3.png"))
        self.canvas.create_image(381.0, 497.0, image=self.image_image_3)

        # Status Label
        self.status_label = Label(
            self.window, 
            text="", 
            bg="#FFFFFF", 
            fg="black"
        )
        self.status_label.place(x=250, y=550)

        # Buttons
        self.button_image_1 = PhotoImage(file=relative_to_assets("button_1.png"))
        self.button_1 = Button(
            image=self.button_image_1,
            borderwidth=0,
            highlightthickness=0,
            command=self.window.quit,
            relief="flat"
        )
        self.button_1.place(
            x=374.0,
            y=254.0,
            width=344.0,
            height=127.0
        )

        self.button_image_2 = PhotoImage(file=relative_to_assets("button_2.png"))
        self.button_2 = Button(
            image=self.button_image_2,
            borderwidth=0,
            highlightthickness=0,
            command=self.process_sentence,
            relief="flat"
        )
        self.button_2.place(
            x=82.0,
            y=261.0,
            width=292.0,
            height=116.0
        )

        # Video-related attributes
        self.video_queue = []
        self.video_folder = r"assets1"  # Update this path

    def on_entry_click(self, event):
        """Function to be called when entry is clicked"""
        if self.entry.get() == "Enter sentence here...":
            self.entry.delete(0, "end")
            self.entry.config(fg='black')

    def on_focusout(self, event):
        """Function to be called when entry loses focus"""
        if self.entry.get() == "":
            self.entry.insert(0, "Enter sentence here...")
            self.entry.config(fg='#888888')

    def process_sentence(self):
        # Use .get() method to retrieve text from Entry widget
        sentence = self.entry.get().strip()
        
        # Ignore placeholder text
        if sentence == "Enter sentence here...":
            messagebox.showerror("Error", "Please enter a sentence!")
            return
        
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
            self.status_label.config(text=f"Processing: {sentence}")
            threading.Thread(target=self.play_videos, daemon=True).start()

    def add_letters_to_queue(self, word):
        for letter in word:
            if self.video_exists(letter):
                self.video_queue.append(letter)
    
    def video_exists(self, name):
        return os.path.exists(os.path.join(self.video_folder, f"{name}.mp4"))
    
    def play_videos(self):
        try:
            for video_name in self.video_queue:
                video_path = os.path.join(self.video_folder, f"{video_name}.mp4")
                
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
            self.status_label.config(text="Playback Complete")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")

def main():
    window = Tk()
    app = SignLanguageApp(window)
    window.mainloop()

if __name__ == "__main__":
    main()