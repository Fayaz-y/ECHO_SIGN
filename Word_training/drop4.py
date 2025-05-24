import cv2
import dropbox
import time
import serial
from datetime import datetime
import keys
from twilio.rest import Client

# Load secure Dropbox token
dbx = dropbox.Dropbox(keys.dropbox_access_token)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
video_filename = f"captured_video_{timestamp}.mp4"
video_path = f"./{video_filename}"
dropbox_path = f"/videos/{video_filename}"

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_size = (640, 480)  # Set fixed frame size

out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

start_time = time.time()
while int(time.time() - start_time) < 10:  # Record for 10 seconds
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

    # Show video feed in a window
    cv2.imshow("Recording...", frame)

    # Press 'q' to stop early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("🎥 Video Captured Successfully!")

# Upload File to Dropbox
with open(video_path, "rb") as file:
    dbx.files_upload(file.read(), dropbox_path, mode=dropbox.files.WriteMode("overwrite"))

print("✅ File uploaded to Dropbox!")

# Create a Shareable Link
shared_link_metadata = dbx.sharing_create_shared_link_with_settings(dropbox_path)
dropbox_link = shared_link_metadata.url

print("🔗 Dropbox File Link:", dropbox_link)

# Send SMS using Twilio
client = Client(keys.account_sid, keys.account_token)

message = client.messages.create(
    from_=keys.twilio_number,
    body=f"""Emergency alert
    Location link: http://maps.google.com/maps?q=loc:11.077632,77.142873
    Video link: {dropbox_link}""",
    to=keys.my_phone_number
)

print(f"📩 SMS sent! Message SID: {message.sid}")
