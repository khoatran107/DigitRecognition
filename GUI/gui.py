import tkinter as tk
import numpy as np

def get_grayscale_array(canvas):
    # Get the drawn content from the canvas
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Take a screenshot of the canvas area
    image = ImageGrab.grab().crop((x, y, x1, y1))

    # Resize the image to 28x28 and convert to grayscale
    image = image.resize((28, 28), Image.ANTIALIAS).convert('L')

    # Convert the image to a numpy array
    grayscale_array = np.array(image)

    # Normalize the values between 0 and 1
    grayscale_array = grayscale_array / 255.0

    return grayscale_array

def on_canvas_button_press(event):
    global prev_x, prev_y
    prev_x = event.x
    prev_y = event.y

def on_canvas_drag(event):
    global prev_x, prev_y
    x = event.x
    y = event.y
    canvas.create_line(prev_x, prev_y, x, y, width=10, capstyle=tk.ROUND, smooth=tk.TRUE, fill='black')
    prev_x = x
    prev_y = y

def process_image():
    grayscale_array = get_grayscale_array(canvas)
    print("Grayscale 2D Array:")
    print(grayscale_array)

root = tk.Tk()
root.title("Draw Digit")

canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack()

canvas.bind("<Button-1>", on_canvas_button_press)
canvas.bind("<B1-Motion>", on_canvas_drag)

process_button = tk.Button(root, text="Process Image", command=process_image)
process_button.pack()

root.mainloop()
