from cv2 import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import pytesseract 

def convertImg2Binary(img, dilate=True):
    # converting img to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # applying gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (9, 9), 0)

    # converting grayscale image to binary image using adaptive thresholding
    thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2)
    
    # applying dilation to thicken the borders
    if dilate:
        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
        proc = cv2.dilate(thres, kernel)
        return proc
    
    return thres

def findSudoku(img):
    # find all the contours and extract the contour with max area
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    board_contour = max(contours, key=cv2.contourArea)

    # finding all the vertices of the sudoku puzzle
    board_contour_points = [contour[0] for contour in board_contour]
    top_left = min(board_contour_points, key= lambda x: x[0] + x[1])
    top_right = max(board_contour_points, key= lambda x: x[0] - x[1])
    bottom_left = min(board_contour_points, key= lambda x: x[0] - x[1])
    bottom_right = max(board_contour_points, key= lambda x: x[0] + x[1])

    # converting vertices to numpy float32 array
    vertices = np.array([top_left, top_right, bottom_left, bottom_right])

    return vertices

def plot_sudoku_vertices(img, vertices):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.circle(img_rgb, tuple(vertices[0]), 10, (255, 0, 0), -1)
    cv2.circle(img_rgb, tuple(vertices[1]), 10, (255, 0, 0), -1)
    cv2.circle(img_rgb, tuple(vertices[2]), 10, (255, 0, 0), -1)
    cv2.circle(img_rgb, tuple(vertices[3]), 10, (255, 0, 0), -1)

    return img_rgb   

def applyTransformations(img):
    proc = convertImg2Binary(img)

    # finding the top-down perspective of the puzzle
    vertices = findSudoku(proc)
    vertices = vertices.astype("float32")
    new_vertices = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

    # converting vertices into new vertices
    matrix = cv2.getPerspectiveTransform(vertices, new_vertices)
    result = cv2.warpPerspective(img, matrix, (450, 450))

    return result

def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv2.resize(img, (w, h))
	img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
	return cv2.resize(img, (size, size))

def findDigits(img, puzzle_size=9):
    res_size = 28
    row_size = int(450 / puzzle_size)
    col_size = int(450 / puzzle_size)
    
    binary_img = convertImg2Binary(img, False)
    height, width = img.shape[:2]

    result = np.zeros((puzzle_size*res_size, puzzle_size*res_size), np.uint8)
    
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            digit = binary_img[i*row_size:(i+1)*row_size, j*col_size:(j+1)*col_size]
            margin = int(np.mean([row_size, col_size]) / 2.5)

            top_left = (margin, margin)
            bottom_right = (row_size - margin, col_size - margin)

            max_area = 0
            seed_point = (None, None)
            digit_rect = None

            # searches for digits in a small square in the middle of the box
            for x in range(top_left[0], bottom_right[0]):
                for y in range(top_left[1], bottom_right[1]):
                    if digit[y, x] == 255 and x < row_size and y < col_size:
                        area, _, _, rect = cv2.floodFill(digit, None, (x, y), 64)
                        if area > max_area:
                            max_area = area
                            seed_point = (x, y)
                            digit_rect = rect

            # converts remaining white pixels to gray
            for x in range(row_size):
                for y in range(col_size):
                    if digit[y, x] == 255 and x < row_size and y < col_size:
                        area = cv2.floodFill(digit, None, (x, y), 64)
            
            # highlights the digit if exists
            mask = np.zeros((row_size + 2, col_size + 2), np.uint8)
            if all([p is not None for p in seed_point]):
                cv2.floodFill(digit, mask, seed_point, 255)
            
            # convert all the remaining pixels back to black
            for x in range(row_size):
                for y in range(col_size):
                    if digit[y, x] == 64 and x < row_size and y < col_size:
                        cv2.floodFill(digit, mask, (x, y), 0)

            if digit_rect is not None:
                x, y, w, h = digit_rect
                digit = digit[y:y+h, x:x+w]
                if w > 0 and h > 0 and (w*h) > 100 and len(digit):
                    res = scale_and_centre(digit, res_size, 4)
                else:
                    res = np.zeros((res_size, res_size))
            else:
                res = np.zeros((res_size, res_size)) 

            res = cv2.bitwise_not(res)
            result[i*res_size:(i + 1)*res_size, j*res_size:(j + 1)*res_size] = res

    return result

def solveSudoku(img, puzzle_size=9):
    res_size = 28
    model = load_model("model.h5")
    result = np.zeros((puzzle_size, puzzle_size))
    mask = np.zeros((res_size, res_size))
    for i in range(puzzle_size):
        for j in range(puzzle_size):
            digit = img[i*res_size:(i+1)*res_size, j*res_size:(j+1)*res_size]
            checksum = mask + digit
            if np.sum(checksum) <= 10:
                continue

            digit = digit.reshape(1, res_size, res_size, 1)
            digit = digit.astype("float32")
            digit /= 255
            res = model.predict(digit)
            result[i][j] = res.argmax()
    
    return result
