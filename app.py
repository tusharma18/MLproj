import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from keras.models import load_model
import time

# Load the trained model
classifier = load_model('model.h5')

# Constants for image size
image_x, image_y = 64, 64

box_x, box_y, box_width, box_height = 100, 100, 300, 300

# Initialize variables
append_text = ''
finalBuffer = []
z = 0

recognized_start_time = 0
recognized_duration = 0
recognized_gesture = ''

sentence_formation_active = False


def predictor():
    import numpy as np
    from keras.preprocessing import image
    test_image = image.load_img('1.png', target_size=(64, 64), color_mode='grayscale')
    # test_image = test_image.convert('RGB')
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)

    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    if result.max() > 0.8:
        index = np.argmax(result)
        return alphabet[index]
    else:
        return ''


def update_trackbars():
    l_h = l_h_var.get()
    l_s = l_s_var.get()
    l_v = l_v_var.get()
    u_h = u_h_var.get()
    u_s = u_s_var.get()
    u_v = u_v_var.get()

    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Crop the region of interest
    x, y, w, h = box_x, box_y, box_width, box_height
    imcrop = frame[y:y + h, x:x + w]

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Update mask display
    mask_image = Image.fromarray(mask)
    mask_photo = ImageTk.PhotoImage(mask_image)
    mask_label.config(image=mask_photo)
    mask_label.image = mask_photo

    # Save the mask image
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)

    # Update the recognized letter
    img_text = predictor()
    scanned_label.config(text=img_text)

    ## Update the sentence forming
    global append_text
    global z
    global recognized_start_time
    global recognized_duration
    global recognized_gesture
    global sentence_formation_active

    if sentence_formation_active:   
        if img_text != '':
            if recognized_gesture == img_text:
                recognized_duration = time.time() - recognized_start_time
            else:
                recognized_gesture = img_text
                recognized_start_time = time.time()
                recognized_duration = 0

            if recognized_duration >= 3:
                append_text += recognized_gesture
                recognized_gesture = ''
                recognized_duration = 0
    
    # Update the scanned label
    scanned_label.config(text=img_text)

    sentence_text.delete('1.0', tk.END)
    sentence_text.insert(tk.END, append_text)

def toggle_sentence_formation():
    global sentence_formation_active
    sentence_formation_active = not sentence_formation_active
    if sentence_formation_active:
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)
    else:
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

def toggle_freeze(event):
    global append_text
    if event.char == 'c':
        append_text += scanned_label.cget('text')
        sentence_text.delete('1.0', tk.END)
        sentence_text.insert(tk.END, append_text)

def update_append_text(event):
    global append_text
    append_text = sentence_text.get('1.0', tk.END).strip()


def save_final_buffer():
    with open("temp.txt", "w") as f:
        f.write('\n'.join(finalBuffer))

# Create the main window
root = tk.Tk()
root.title("ASL Recognition")
root.geometry("800x600")

# Create main frame
main_frame = tk.Frame(root)
main_frame.pack()

# Create top frame for video stream, template, trackbars, and mask
top_frame = tk.Frame(main_frame)
top_frame.pack(side=tk.TOP)

# Create bottom frame for buttons and sentence box
bottom_frame = tk.Frame(main_frame)
bottom_frame.pack(side=tk.BOTTOM)

# Create video frame
video_frame = tk.Frame(top_frame)
video_frame.pack(side=tk.LEFT)

# Create template frame
template_frame = tk.Frame(top_frame)
template_frame.pack(side=tk.LEFT)

# Create trackbars frame
trackbars_frame = tk.Frame(top_frame)
trackbars_frame.pack(side=tk.LEFT)

# Create mask frame
mask_frame = tk.Frame(bottom_frame)
mask_frame.pack(side=tk.LEFT)

# Create sentence forming frame
sentence_frame = tk.Frame(bottom_frame)
sentence_frame.pack(side=tk.LEFT)

# Create scanned label
scanned_label = tk.Label(sentence_frame, text="", font=("Arial", 48))
scanned_label.pack(pady=20)

# Create sentence text box
sentence_text = tk.Text(sentence_frame, width=40, height=5, font=("Arial", 36))
sentence_text.pack()

# Create buttons frame
buttons_frame = tk.Frame(bottom_frame)
buttons_frame.pack(side=tk.RIGHT)

# Create start and stop buttons
start_button = tk.Button(buttons_frame, text="Start", command=toggle_sentence_formation)
start_button.pack(pady=10)

stop_button = tk.Button(buttons_frame, text="Stop", command=toggle_sentence_formation, state=tk.DISABLED)
stop_button.pack(pady=10)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create video canvas
video_canvas = tk.Canvas(video_frame, width=640, height=480)
video_canvas.pack()

# Load and resize the template image
template_image = Image.open('template.png')
template_image = template_image.resize((355, 480), Image.ANTIALIAS)
template_photo = ImageTk.PhotoImage(template_image)

# Create template label
template_label = tk.Label(template_frame, image=template_photo)
template_label.pack()

# Create trackbars
l_h_var = tk.IntVar(value=0)
l_s_var = tk.IntVar(value=0)
l_v_var = tk.IntVar(value=0)
u_h_var = tk.IntVar(value=179)
u_s_var = tk.IntVar(value=255)
u_v_var = tk.IntVar(value=255)

l_h_label = tk.Label(trackbars_frame, text="L - H")
l_s_label = tk.Label(trackbars_frame, text="L - S")
l_v_label = tk.Label(trackbars_frame, text="L - V")
u_h_label = tk.Label(trackbars_frame, text="U - H")
u_s_label = tk.Label(trackbars_frame, text="U - S")
u_v_label = tk.Label(trackbars_frame, text="U - V")

l_h_scale = tk.Scale(trackbars_frame, variable=l_h_var, from_=0, to=179, orient=tk.HORIZONTAL, length=200)
l_s_scale = tk.Scale(trackbars_frame, variable=l_s_var, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
l_v_scale = tk.Scale(trackbars_frame, variable=l_v_var, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
u_h_scale = tk.Scale(trackbars_frame, variable=u_h_var, from_=0, to=179, orient=tk.HORIZONTAL, length=200)
u_s_scale = tk.Scale(trackbars_frame, variable=u_s_var, from_=0, to=255, orient=tk.HORIZONTAL, length=200)
u_v_scale = tk.Scale(trackbars_frame, variable=u_v_var, from_=0, to=255, orient=tk.HORIZONTAL, length=200)

l_h_label.pack()
l_h_scale.pack()
l_s_label.pack()
l_s_scale.pack()
l_v_label.pack()
l_v_scale.pack()
u_h_label.pack()
u_h_scale.pack()
u_s_label.pack()
u_s_scale.pack()
u_v_label.pack()
u_v_scale.pack()

# Create mask display
mask_label = tk.Label(mask_frame)
mask_label.pack()

# Bind key events
root.bind('<Key>', toggle_freeze)
sentence_text.bind('<KeyRelease>', update_append_text)

# Update the UI
def update():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.rectangle(frame_rgb, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 2)

    # Resize the frame for display
    img = cv2.resize(frame_rgb, (640, 480))
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_canvas.imgtk = imgtk
    video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

    # Process trackbar updates
    update_trackbars()

    # Call this function again after 10 milliseconds
    video_canvas.after(100, update)

# Start the video update process
update()

# Start the main event loop
root.mainloop()
