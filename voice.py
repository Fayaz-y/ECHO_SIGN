import speech_recognition as sr
import subprocess

recognizer = sr.Recognizer()

# Replace 1 with your actual mic index
mic_index = 1

with sr.Microphone(device_index=mic_index) as source:
    print("🎤 Speak using webcam mic...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print("📝 You said:", text)

        # Convert the sentence into a list of words
        word_list = text.split()
        print("📋 Word List:", word_list)

        # Check if 'help' is in the word list (case-insensitive)
        if "help" in [word.lower() for word in word_list]:
            subprocess.Popen(["python", "code_sms.py"])
            alert_sent = True

            # Place your alert logic here
            # Example: send_alert_notification()

    except sr.UnknownValueError:
        print("😕 Could not understand the audio.")
    except sr.RequestError as e:
        print(f"🔌 API error: {e}")
