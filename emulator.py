from ppadb.client import Client as ADBClient

from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from sudoku import Sudoku

client = ADBClient()
devices = client.devices()

class Utils:
    
    @staticmethod
    def get_screen(device_id=0):

        if len(devices) <= device_id:
            print("No devices connected")
            quit()

        device = devices[device_id]
        screen_buffer = device.screencap()

        with open("screen.png", "wb") as f:
            f.write(screen_buffer)
        screen_array = np.asarray(screen_buffer, dtype=np.uint8)

        screen = cv2.imdecode(screen_array, cv2.IMREAD_GRAYSCALE)

        return screen
    
    @staticmethod
    def convertImg2Binary(img, dilate=False):
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
    
    @staticmethod
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

    @staticmethod
    def plot_sudoku_vertices(img, vertices):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.circle(img_rgb, tuple(vertices[0]), 10, (255, 0, 0), -1)
        cv2.circle(img_rgb, tuple(vertices[1]), 10, (255, 0, 0), -1)
        cv2.circle(img_rgb, tuple(vertices[2]), 10, (255, 0, 0), -1)
        cv2.circle(img_rgb, tuple(vertices[3]), 10, (255, 0, 0), -1)

        return img_rgb
    
    @staticmethod
    def applyTransformations(img):
        proc = Utils.convertImg2Binary(img)

        # finding the top-down perspective of the puzzle
        vertices = Utils.findSudoku(proc)
        vertices = vertices.astype("float32")
        new_vertices = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

        # converting vertices into new vertices
        matrix = cv2.getPerspectiveTransform(vertices, new_vertices)
        result = cv2.warpPerspective(img, matrix, (450, 450))

        return result

    @staticmethod
    def get_sudoku_matrix(sudoku, size=50):
        result = []
        for row in range(9):
            row_res = []
            for col in range(9):
                digit = sudoku[row*size + 2:(row + 1)*size - 2, col*size + 2:(col + 1)*size - 2]
                ocr_result = pytesseract.image_to_string(digit, lang='eng',
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                if ocr_result[0] not in [str(i) for i in range(10)]:
                    row_res.append(0)
                else:
                    row_res.append(int(ocr_result[0]))
            result.append(row_res)
        return result

def fill_sudoku_board(unsolved_board, solved_board):
    origin_x, origin_y = (12, 465)

    locations = {
        1: (100, 2050),
        2: (200, 2050),
        3: (300, 2050),
        4: (400, 2050),
        5: (500, 2050),
        6: (600, 2050),
        7: (750, 2050),
        8: (850, 2050),
        9: (950, 2050)
    }
    
    for row in unsolved_board:
        print(row)

    for row in solved_board:
        print(row)
    
    device = devices[0]
    for i in range(9):
        for j in range(9):
            if unsolved_board[i][j] != solved_board[i][j]:
                cell_x = origin_x + (j * 120) + 60
                cell_y = origin_y + (i * 120) + 60

                btn_x, btn_y = locations[solved_board[i][j]]
                
                device.shell(f"input tap {cell_x} {cell_y}")
                device.shell(f"input tap {btn_x} {btn_y}")

screen = Utils.get_screen()
sudoku_img = Utils.applyTransformations(screen)
unsolved_board = Utils.get_sudoku_matrix(sudoku_img)

board = Sudoku(3, 3, board=unsolved_board)
solution = board.solve()
solved_board = solution.board

fill_sudoku_board(unsolved_board, solved_board)

# devices[0].shell("input tap 192 477")
# plt.imshow(screen[465:1500, 12:-13], cmap="gray")
# plt.show()

# y = 2050


# screen = cv2.imread("screen.png", cv2.IMREAD_GRAYSCALE)
# board = Sudoku(3, 3, board=result)
# solution = board.solve()
# solution.show()

# print(solution.board[0])
# for row in result:
#     print(row)