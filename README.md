# 🤟 ECHO_SIGN

**ECHO_SIGN** is an assistive application developed for people with speech and hearing disabilities. It enables seamless communication through **Sign-to-Text**, **Text-to-Sign**, and an emergency **SOS alert** feature. The system uses **TensorFlow**, **MediaPipe**, and a CNN-based model for real-time hand gesture recognition.

---

## 🧠 Features

- 🔤 **Sign to Text** – Converts real-time sign language into text using a CNN model.
- 📄 **Text to Sign** – Converts text into sign language gestures using animated representations.
- 🚨 **SOS Alert** – Sends an emergency SMS to a predefined number using GSM/SIM module.
- 🗣️ **Sign to Speech** *(Coming Soon)* – Converts recognized sign language into spoken words.

---

## 🛠 Tech Stack

| Category       | Tools Used                         |
|----------------|------------------------------------|
| Language       | Python                             |
| ML Frameworks  | TensorFlow, Keras                  |
| CV & Gesture   | MediaPipe, OpenCV                  |
| Interface      | Tkinter                            |
| Model Type     | CNN (Convolutional Neural Network) |
| Dataset        | Custom hand sign dataset           |
| File Format    | `.h5` for CNN model                |

---

## 📁 Project Structure

```
ECHO_SIGN/
│
├── AtoZ/                           # Static sign image resources
├── LSTM_model_for_detect_words/    # Placeholder for future LSTM model
├── Word_training/                  # Training scripts for hand signs
├── assets_base/, assets_text/      # Frame resources for animations
│
├── base.py                         # Main entry point (launches UI)
├── cnn8grps_rad1_model.h5         # Trained CNN model
├── dataset_metadata.json          # Metadata for gesture labels
├── interface.py                    # UI handler for options
├── sign_to_text.py                # Sign language to text
├── text_to_sign.py                # Text to animated signs
├── sms.py / Code_sms.py           # SOS feature implementation
├── voice.py                       # Sign to speech (WIP)
├── keys.py                        # Configuration keys
├── spell.py                       # Internal logic files
├── test.txt                       # Testing data
├── logo.png, white.jpg            # UI assets
└── README.md                      # This documentation
```

---

## 🚀 Getting Started

### 🔧 Prerequisites

Ensure Python 3.7 or higher is installed.

Install all required dependencies:

```bash
pip install tensorflow opencv-python mediapipe pillow
```

Or use requirements.txt if available:

```bash
pip install -r requirements.txt
```

### ⚙️ Configuration Setup

Create a `keys.py` file in the root directory with the following structure:

```python
# keys.py file setup
account_sid = "twilio_sid"
account_token = "twilio_token" 
twilio_number = "+twilio_no"
my_phone_number = "+your_ph_no"
dropbox_access_token = "put_generated_token"
```

Replace the placeholder values with your actual credentials:
- Get Twilio credentials from [Twilio Console](https://console.twilio.com/)
- Generate Dropbox access token from [Dropbox App Console](https://www.dropbox.com/developers/apps)

### ▶️ Run the Application

Launch the main program using:

```bash
python base.py
```

A GUI will open with three main features:
- Sign to Text
- Text to Sign  
- SOS Emergency Alert

**Note:** Ensure your webcam is working and has proper lighting for accurate sign detection.

---

## 💻 Development Setup (VS Code)

### 1. Clone the Repository

```bash
git clone https://github.com/Fayaz-y/Women-Safety-Web-dashboard.git
cd Women-Safety-Web-dashboard
```

### 2. Open in VS Code

```bash
code .
```

### 3. Install Python Extension

Install the official Python extension for VS Code for better development experience.

### 4. Set Up Virtual Environment (Recommended)

```bash
python -m venv echo_sign_env
source echo_sign_env/bin/activate  # On Windows: echo_sign_env\Scripts\activate
pip install -r requirements.txt
```

---

## 🧠 Model Overview

- **Model Type:** CNN (Convolutional Neural Network)
- **File:** `cnn8grps_rad1_model.h5`
- **Input:** Hand sign image captured in real-time
- **Output:** Alphabet/word classification

The model detects and identifies hand gestures corresponding to letters using a custom dataset.

---

## 🚧 Roadmap

- ✅ Sign to Text
- ✅ Text to Sign
- ✅ SOS Emergency
- 🔄 Sign to Speech (Under Development)

---

## 📬 Contributing

Contributions are welcome! You can fork the repository and create pull requests to:

- Add more sign gestures
- Improve model accuracy
- Complete speech integration
- Enhance UI/UX
- Add documentation

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ✨ Acknowledgements

- Developed as a social good project for differently-abled individuals
- Powered by TensorFlow, MediaPipe, and OpenCV
- Thanks to the open-source community for tools and inspiration
- Special thanks to all contributors and testers

---

## 📞 Support

If you encounter any issues or have questions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Made with ❤️ for accessibility and inclusion**
