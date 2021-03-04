import streamlit as st
from cv2 import cv2 
import numpy as np
from sudoku import *

st.title("Sudoku Solver using OpenCV, Deep Learning CNN using Python, Tensorflow and Keras")
uploaded_file = st.file_uploader("Pick a file")

if uploaded_file is not None:
    st.sidebar.title("Filters")
    st.sidebar.text("")
    show_steps = st.sidebar.checkbox("Show Steps", True)
    show_results = st.sidebar.checkbox("Show Results", True)
    show_code = st.sidebar.checkbox("Show Code", True)
    file_buffer = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_buffer, 1)
    st.image(img, channels="BGR") 

    st.header("Converting Image from RGB to Binary")

    if show_steps:
        st.write("""
            > converting image to binary makes it easy to detect edges
            
            ### steps taken
            1. Converted image from rgb to grayscale.
            2. Applied Gaussian Blur to reduce noise in the image.
            3. Applied thresholding to convert image to binary image.
            4. Here instead of using Global Thresholding I have used Adaptive thresholding to get better results.
            5. Finally applied Dilation to broaden the outlines.
        """)

    if show_code:
        st.write("""
        ```python
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
        ```
        """)
    transformed_img = convertImg2Binary(img)

    if show_results:
        st.image(transformed_img)

    st.header("Finding the vertices of the sudoku board")
    if show_steps:
        st.write("""
            ### steps taken
            1. Found all the external countours
            2. Assuming puzzle will be main object of the image, the countour with maximum area gives the board contour.
            3. After finding the contour now it turn to calculate the vertices. For calculating vertices we can use following logic:
                * __TOP LEFT__ (x is min, y is min) => (x + y) is min
                * __TOP RIGHT__ (x is max, y is min) => (x - y) is max
                * __BOTTOM LEFT__ (x is min, y is max) => (y - x) is max
                * __BOTTOM RIGHT__ (x is max, y is max) => (x + y) is max
            4. We plot the obtained vertices on to the image using __opencv circle__ function.
        """)
    
    if show_code:
        st.write("""
            ```python
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
            ```
        """)

    vertices = findSudoku(transformed_img)

    if show_results:
        st.image(plot_sudoku_vertices(transformed_img, vertices))
    
    st.header("Getting the top down view of the sudoku puzzle")

    if show_steps:
        st.write("""
            Now as we already have a idea about the location of vertices of the image, if we could get the top down view of the puzzle we can process the image more easily.

            ### steps taken

            1. We use __OpenCV's getPerspectiveTransform__ to convert points from plane to another
            2. We use __OpenCV's wrapPerspective__ to wrap the image to the points we got from above
        """)
    
    if show_code:
        st.write("""
            ```python
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
            ```
        """)

    top_down_img = applyTransformations(img)
    if show_results:
        st.image(top_down_img) 
        st.image(convertImg2Binary(top_down_img, False))

    st.header("Extracting the digits")
    if show_steps:
        st.write("""
            Now the important part is to extract the digits.

            ### approach
            * From the results obtained from the previous step we can divide the image into 81 equal boxes and start using __OCR__ to recognize the digits.
            * But there is a problem with that approach. As we can see when we divide the puzzle into small boxes there will be lot of other pixels in that box and it will become difficult for ocr to recognize.
            * So, we need to clean the boxes before we start to use OCR.

            ### steps taken
            1. We divide the image into 81 (required number of boxes) boxes.
            2. Now we assume small region at the centre of each box and see if we have any white pixels, assuming the pixel pixels belongs to some digits.
            3. In that small region, using __OpenCV's floodFill__ function we find the largest connected pixel area and if that area is above a certain threshold we assume it as digit and obtain the seed point.
            4. While searching in that box we fill the largest connected area with gray color.
            5. Now we fill the remaining white pixels (which is noise) with gray color.
            6. Using the seed point (if exists) we fill the digit (largest connected area) with white pixels and remaining gray pixel with black.
            7. Then we resize and apply padding to those boxes to make sure they match the dimensions of the test data we will train our __CNN__ model with.
        """)
    
    if show_code:
        st.write("""
        ```python
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

                    res = cv2.resize(digit, (res_size, res_size))
                    
                    result[i*res_size:(i + 1)*res_size, j*res_size:(j + 1)*res_size] = res

            return result
        ```
    """)

    result = findDigits(top_down_img)
    if show_results:
        st.image(result)

    st.header("Creating and training CNN model to detect the digits")
    if show_steps:
        st.write("""
            # Now we detect the digits using our trained CNN model
            
            ### steps taken

            1. Used Keras Sequential Model
            2. Convolution Layer + Max Pooling Layer + Flatter Layer + Dense
            
            > I trained my model with MNIST dataset for 20 epochs and attained a accuracy of 98%
        """)
    
    if show_code:
        st.write("""
        ```python
            import tensorflow as tf
            import matplotlib.pyplot as plt
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
            input_shape = (28, 28, 1)

            x_train = x_train.astype("float32")
            x_test = x_test.astype("float32")

            x_train /= 255
            x_test /= 255

            model = Sequential()
            model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
            model.add(Dense(128, activation=tf.nn.relu))
            model.add(Dropout(0.2))
            model.add(Dense(10,activation=tf.nn.softmax))

            model.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
            model.fit(x=x_train,y=y_train, epochs=20)

            model.evaluate(x_test, y_test)

            model.save("model.h5")
        ```
    """)

    st.header("Detecting the digits using our trained CNN model")
    if show_code:
        st.write("""
        ```python
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
        ``` 
    """)

    result = solveSudoku(result)

    for i in range(9):
        st.write(f"{result[i]}")

    st.write("""
    > Result obtained may not be perfect but with more training of model with more datasets can produce better results.
    """)

