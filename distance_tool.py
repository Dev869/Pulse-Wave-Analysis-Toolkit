#!/usr/bin/env python3
import sys
import math
import tkinter as tk
from PIL import Image, ImageTk

# Sequential picker: PROXIMAL first, then DISTAL

def pick_point(image_path, title):
    root = tk.Tk()
    root.title(title)

    img = Image.open(image_path)
    tk_img = ImageTk.PhotoImage(img)
    canvas = tk.Canvas(root, width=img.width, height=img.height)
    canvas.pack()
    canvas.create_image(0, 0, anchor='nw', image=tk_img)

    clicked = []
    def on_click(event):
        x, y = event.x, event.y
        r = 5
        canvas.create_oval(x-r, y-r, x+r, y+r, outline='red', width=2)
        clicked.append((x, y))
        root.destroy()
    canvas.bind('<Button-1>', on_click)
    root.mainloop()
    return clicked[0] if clicked else None

if __name__ == '__main__':
    # Expect PROXIMAL then DISTAL
    if len(sys.argv) != 3:
        print('Usage: python distance_tool.py proximal_image.png distal_image.png')
        sys.exit(1)

    prox_path, dist_path = sys.argv[1], sys.argv[2]

    # 1) Pick on PROXIMAL image
    pt1 = pick_point(prox_path, 'Select point on PROXIMAL image')
    if pt1 is None:
        print('Error: no point selected on PROXIMAL image', file=sys.stderr)
        sys.exit(2)

    # 2) Pick on DISTAL image
    pt2 = pick_point(dist_path, 'Select point on DISTAL image')
    if pt2 is None:
        print('Error: no point selected on DISTAL image', file=sys.stderr)
        sys.exit(3)

    # Compute pixel distance
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist_pixels = math.hypot(dx, dy)
    print(f'Pixel distance: {dist_pixels:.2f}')