# Copyright by EinS0ne. All rights reserved. Completed: January 8, 2025.
import cv2
import mediapipe as mp
import numpy as np
import pygame
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("VisionXO")
font = pygame.font.Font(None, 36)
GRID_SIZE = 20  # Số ô lưới trên màn hình
CELL_SIZE = 40  # Kích thước mỗi ô
board = [[" " for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
player_turn = "X"
last_player_hand = False
game_over = False
winning_cells = []
highlighted_cell = None

def draw_grid():
    """Vẽ bảng lưới và các ô đã điền"""
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x, y = col * CELL_SIZE, row * CELL_SIZE
            color = (200, 200, 200)
            if (row, col) in winning_cells:  # Bôi đỏ các ô thắng (chỉnh màu ở color)
                color = (180, 0, 0)
            elif highlighted_cell == (row, col):  # Bôi đỏ ô đang trỏ (chỉnh màu ở color)
                color = (180, 0, 0)
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), 4)
            if board[row][col] != " ":
                text = font.render(board[row][col], True, (0, 0, 0))
                screen.blit(text, (x + CELL_SIZE // 4, y + CELL_SIZE // 4))

def check_fingers(hand_landmarks):
    """Kiểm tra trạng thái ngón tay"""
    fingers_up = [False, False]
    if hand_landmarks:
        lm = hand_landmarks.landmark
        fingers_up[0] = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < lm[mp_hands.HandLandmark.INDEX_FINGER_DIP].y
        fingers_up[1] = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < lm[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y
    return fingers_up

# Hàm kiểm tra các ô xem có thắng hay chưa
def check_winner(player):
    """Kiểm tra người chơi thắng"""
    global winning_cells
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if board[row][col] == player:
                if col + 4 < GRID_SIZE and all(board[row][col + i] == player for i in range(5)):
                    winning_cells = [(row, col + i) for i in range(5)]
                    return True
                if row + 4 < GRID_SIZE and all(board[row + i][col] == player for i in range(5)):
                    winning_cells = [(row + i, col) for i in range(5)]
                    return True
                if row + 4 < GRID_SIZE and col + 4 < GRID_SIZE and all(board[row + i][col + i] == player for i in range(5)):
                    winning_cells = [(row + i, col + i) for i in range(5)]
                    return True
                if row - 4 >= 0 and col + 4 < GRID_SIZE and all(board[row - i][col + i] == player for i in range(5)):
                    winning_cells = [(row - i, col + i) for i in range(5)]
                    return True
    return False

cap = cv2.VideoCapture(0)
running = True
while running:
    screen.fill((255, 255, 255))
    draw_grid()
    pygame.display.flip()
    success, img = cap.read()
    if not success or game_over:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    highlighted_cell = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = check_fingers(hand_landmarks)
            if fingers == [True, False]:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * GRID_SIZE), int(index_finger_tip.y * GRID_SIZE)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    highlighted_cell = (y, x)
            if fingers == [True, True] and not last_player_hand:
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * GRID_SIZE), int(index_finger_tip.y * GRID_SIZE)
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and board[y][x] == " ":
                    board[y][x] = player_turn
                    if check_winner(player_turn):
                        game_over = True
                    player_turn = "O" if player_turn == "X" else "X"
                    last_player_hand = True
            elif fingers != [True, True]:
                last_player_hand = False

cv2.imshow("Camera", img)
if cv2.waitKey(1) & 0xFF == ord('q'):
    running = False
cap.release()
cv2.destroyAllWindows()
pygame.quit()
