# demo.py
# pac-man game controlled by hand gestures using trained HMM models
# thumbs up    -> move up
# point right  -> move right
# point left   -> move left
# thumbs down  -> move down
# peace sign   -> toggle pause (hold for about 1.5 seconds)
# open palm    -> reset pac-man to starting position

import cv2
import numpy as np
import pickle
import time
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision
from collections import deque

MODEL_PATH = "hand_landmarker.task"

# gesture order must match features.py and train_hmms.py exactly
GESTURE_NAMES = ["up", "right", "left", "down", "peace", "palm"]

COLS = 19
ROWS = 15
CELL = 36

MAZE = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
    [1,0,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1],
    [1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,1,1],
    [1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1],
    [1,0,1,1,0,1,0,1,1,1,1,1,0,1,0,1,1,0,1],
    [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1],
    [1,0,1,0,1,1,1,0,0,1,0,0,1,1,1,0,1,0,1],
    [1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1],
    [1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),(0,5)
]

# same triplets as features.py - must not change
JOINT_TRIPLETS = [
    (0,1,2),(1,2,3),(2,3,4),(0,5,6),(5,6,7),(6,7,8),
    (0,9,10),(9,10,11),(10,11,12),(0,13,14),(13,14,15),(14,15,16),
    (0,17,18),(17,18,19),(18,19,20)
]
SPREAD_TRIPLETS = [(5,0,9),(9,0,13),(13,0,17),(1,0,5)]

# lowered from 15 to 8 - gestures now trigger much more reliably
CONFIDENCE_MARGIN = 5.0

# lowered from 2 to 1 - movement responds on first confident prediction
HOLD_MOVE = 1

# 8 consecutive windows for peace = about 1.5 seconds at 30fps
HOLD_PEACE = 8

PAUSE_COOLDOWN = 2.0
MOVE_INTERVAL  = 0.18


def angle_at(a, b, c):
    # same formula as features.py: angle at joint b between vectors b->a and b->c
    v1 = a - b
    v2 = c - b
    denom   = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
    cos_val = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return np.arccos(cos_val)


def frame_features(lm):
    # compute all 19 angles for one frame of landmarks
    angles = [angle_at(lm[a], lm[b], lm[c]) for a, b, c in JOINT_TRIPLETS + SPREAD_TRIPLETS]
    return np.array(angles)


def draw_hand(frame, lm_list, h, w):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lm_list]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 1)
    for pt in pts:
        cv2.circle(frame, pt, 3, (200, 255, 200), -1)


def is_wall(r, c):
    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return True
    return MAZE[r][c] == 1


def bfs_next(ghost_pos, target_pos):
    # breadth-first search - finds shortest path through maze for ghost AI
    if ghost_pos == target_pos:
        return ghost_pos
    from collections import deque as dq
    queue   = dq([(ghost_pos, [])])
    visited = {ghost_pos}
    while queue:
        pos, path = queue.popleft()
        r, c = pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            npos = (r+dr, c+dc)
            if not is_wall(*npos) and npos not in visited:
                new_path = path + [npos]
                if npos == target_pos:
                    return new_path[0] if new_path else ghost_pos
                visited.add(npos)
                queue.append((npos, new_path))
    return ghost_pos


open_cells = [(r, c) for r in range(ROWS) for c in range(COLS) if MAZE[r][c] == 0]


def fresh_dots():
    d = {pos: 1 for pos in open_cells}
    d[(7, 9)]  = 0
    d[(1, 1)]  = 0
    d[(1, 17)] = 0
    return d


# game state
player_pos = [7, 9]
player_dir = [0, 1]
dot_grid   = fresh_dots()
ghosts = [
    {"pos": [1, 1],  "color": (50, 50, 255),  "timer": 0.0, "speed": 0.7},
    {"pos": [1, 17], "color": (220, 50, 220), "timer": 0.0, "speed": 0.9},
]
score      = 0
lives      = 3
paused     = False
game_over  = False
win        = False
mouth_open = True
mouth_timer = 0


def reset_positions():
    global player_pos, player_dir, mouth_open
    player_pos = [7, 9]
    player_dir = [0, 1]
    mouth_open = True
    ghosts[0]["pos"] = [1, 1]
    ghosts[1]["pos"] = [1, 17]


def draw_game(canvas, ox, oy):
    global mouth_open
    for r in range(ROWS):
        for c in range(COLS):
            x = ox + c * CELL
            y = oy + r * CELL
            if MAZE[r][c] == 1:
                cv2.rectangle(canvas, (x+1,y+1), (x+CELL-1,y+CELL-1), (30,55,160), -1)
                cv2.rectangle(canvas, (x+2,y+2), (x+CELL-2,y+CELL-2), (55,90,210), 1)
            elif dot_grid.get((r,c), 0) == 1:
                cx2 = x + CELL//2
                cy2 = y + CELL//2
                cv2.circle(canvas, (cx2, cy2), 3, (255,215,80), -1)

    pr, pc = player_pos
    px = ox + pc*CELL + CELL//2
    py = oy + pr*CELL + CELL//2
    dr, dc = player_dir
    if dc == 1:    angle = 0
    elif dc == -1: angle = 180
    elif dr == -1: angle = 90
    else:          angle = 270
    mouth_angle = 25 if mouth_open else 5
    cv2.ellipse(canvas, (px,py), (CELL//2-3, CELL//2-3),
                -angle, mouth_angle, 360-mouth_angle, (0,215,255), -1)

    for g in ghosts:
        gr, gc = g["pos"]
        gx = ox + gc*CELL + CELL//2
        gy = oy + gr*CELL + CELL//2
        gc2 = g["color"]
        cv2.ellipse(canvas, (gx,gy-2), (CELL//2-3,CELL//2-3), 0, 180, 360, gc2, -1)
        cv2.rectangle(canvas, (gx-CELL//2+3,gy-2), (gx+CELL//2-3,gy+CELL//2-3), gc2, -1)
        for wave in range(3):
            wx  = gx - CELL//2 + 3 + wave * 7
            pts = np.array([[wx,gy+CELL//2-3],[wx+3,gy+CELL//2-7],[wx+6,gy+CELL//2-3]], np.int32)
            cv2.fillPoly(canvas, [pts], (18,18,30))
        cv2.circle(canvas, (gx-4,gy-4), 3, (255,255,255), -1)
        cv2.circle(canvas, (gx+4,gy-4), 3, (255,255,255), -1)
        cv2.circle(canvas, (gx-3,gy-4), 2, (0,0,180), -1)
        cv2.circle(canvas, (gx+5,gy-4), 2, (0,0,180), -1)


if not os.path.exists(MODEL_PATH):
    print(f"ERROR: {MODEL_PATH} not found in current folder")
    exit(1)

print("loading trained HMM models ...")
with open("hmm_models.pkl", "rb") as f:
    models = pickle.load(f)

expected_features = models[0].means_.shape[1]
print(f"loaded 6 HMM models, feature size: {expected_features} per frame")
print(f"confidence threshold: {CONFIDENCE_MARGIN} (gap between top two log-likelihoods)")
print("gesture mapping: 0=up  1=right  2=left  3=down  4=peace  5=palm")
print("controls: gesture to move | peace hold=pause | palm=reset | q=quit\n")

base_options = mp_tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# sliding window of last 20 frames of feature vectors
# classification runs on every new frame once window is full
feat_window       = deque(maxlen=20)
prev_feats        = None
move_history      = deque(maxlen=HOLD_MOVE)
peace_counter     = 0
last_pause_toggle = 0
last_palm_time    = 0
last_move_time    = 0
fps_times         = deque(maxlen=30)
display_gesture   = "waiting..."
display_margin    = 0.0
display_lls       = None

GAME_W   = COLS * CELL
GAME_H   = ROWS * CELL
CANVAS_W = W + GAME_W + 20
CANVAS_H = max(H, GAME_H + 80)


while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # flip horizontally so it acts like a mirror
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    if result.hand_landmarks:
        lm_list = result.hand_landmarks[0]
        draw_hand(frame, lm_list, H, W)
        lm = np.array([[p.x, p.y, p.z] for p in lm_list])

        # 19 joint angles for this frame
        angles = frame_features(lm)

        # velocity = change in angles from previous frame
        vel = angles - prev_feats if prev_feats is not None else np.zeros_like(angles)
        prev_feats = angles

        # 38-dim feature vector = angles + velocity
        fvec = np.concatenate([angles, vel])
        if expected_features == len(angles):
            fvec = angles
        elif expected_features < len(fvec):
            fvec = fvec[:expected_features]

        feat_window.append(fvec)
    else:
        prev_feats = None
        peace_counter = 0
        move_history.clear()

    now = time.time()

    # classify once we have a full 20-frame window
    if len(feat_window) == 20:
        seq = np.array(feat_window)

        # score all 6 HMMs - this runs the Forward algorithm internally
        log_scores = []
        for m in models:
            try:
                log_scores.append(m.score(seq))
            except Exception:
                log_scores.append(-np.inf)
        log_scores = np.array(log_scores)

        best     = int(np.argmax(log_scores))
        sorted_s = np.sort(log_scores)[::-1]
        margin   = float(sorted_s[0] - sorted_s[1])

        display_gesture = GESTURE_NAMES[best]
        display_margin  = margin
        display_lls     = log_scores

        # only act on prediction if margin exceeds confidence threshold
        confident = margin >= CONFIDENCE_MARGIN

        if confident and best == 4:
            # peace - requires sustained hold to toggle pause
            peace_counter += 1
            move_history.clear()
            if peace_counter >= HOLD_PEACE and (now - last_pause_toggle) > PAUSE_COOLDOWN:
                paused = not paused
                last_pause_toggle = now
                peace_counter = 0

        elif confident and best == 5:
            # palm - instantly reset player position
            peace_counter = 0
            move_history.clear()
            if now - last_palm_time > 1.0:
                player_pos = [7, 9]
                player_dir = [0, 1]
                last_palm_time = now

        elif confident and best in {0,1,2,3} and not paused and not game_over and not win:
            # directional gesture - move player one step
            peace_counter = max(0, peace_counter - 1)
            move_history.append(best)
            if (len(move_history) == HOLD_MOVE
                    and len(set(move_history)) == 1
                    and now - last_move_time > MOVE_INTERVAL):
                direction_map = {0:[-1,0], 1:[0,1], 2:[0,-1], 3:[1,0]}
                wanted = direction_map[best]
                nr = player_pos[0] + wanted[0]
                nc = player_pos[1] + wanted[1]
                if not is_wall(nr, nc):
                    player_dir = wanted
                    player_pos = [nr, nc]
                    if dot_grid.get((nr,nc), 0) == 1:
                        dot_grid[(nr,nc)] = 0
                        score += 10
                    last_move_time = now
                    move_history.clear()
                    if all(v == 0 for v in dot_grid.values()):
                        win = True
        else:
            peace_counter = max(0, peace_counter - 2)

    # mouth animation
    mouth_timer += time.time() - t0
    if mouth_timer > 0.18:
        mouth_open  = not mouth_open
        mouth_timer = 0

    # ghost movement using BFS pathfinding
    for g in ghosts:
        g["timer"] += time.time() - t0
        if g["timer"] >= g["speed"] and not paused and not game_over and not win:
            g["timer"] = 0.0
            nxt = bfs_next(tuple(g["pos"]), tuple(player_pos))
            if not is_wall(*nxt):
                g["pos"] = list(nxt)

    # ghost collision
    for g in ghosts:
        if g["pos"] == player_pos and not game_over:
            lives -= 1
            if lives <= 0:
                game_over = True
            else:
                reset_positions()
                time.sleep(0.4)

    # draw everything
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    cam_y  = (CANVAS_H - H) // 2
    canvas[cam_y:cam_y+H, :W] = frame
    cv2.rectangle(canvas, (W,0), (CANVAS_W,CANVAS_H), (10,10,20), -1)

    game_ox = W + 10
    game_oy = 50
    draw_game(canvas, game_ox, game_oy)

    cv2.putText(canvas, f"SCORE: {score}", (game_ox, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,215,255), 2)
    for li in range(lives):
        lx = game_ox + GAME_W - 25 - li * 22
        cv2.ellipse(canvas, (lx,22), (9,9), 0, 25, 335, (0,215,255), -1)

    g_color = (80,230,120)
    if display_gesture == "peace": g_color = (0,215,255)
    elif display_gesture == "palm": g_color = (255,180,0)
    cv2.putText(canvas, display_gesture, (10, cam_y+32),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, g_color, 2)
    cv2.putText(canvas, f"margin:{display_margin:.1f}  need>{CONFIDENCE_MARGIN}",
                (10, cam_y+56), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130,130,130), 1)

    if display_gesture == "peace" and peace_counter > 0:
        bar_w  = W - 20
        filled = min(int(peace_counter / HOLD_PEACE * bar_w), bar_w)
        cv2.rectangle(canvas, (10, cam_y+H-38), (10+bar_w, cam_y+H-24), (40,40,80), -1)
        cv2.rectangle(canvas, (10, cam_y+H-38), (10+filled, cam_y+H-24), (0,215,255), -1)
        action = "unpause" if paused else "pause"
        cv2.putText(canvas, f"hold peace to {action}... {int(peace_counter/HOLD_PEACE*100)}%",
                    (10, cam_y+H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,215,255), 1)

    if display_lls is not None:
        ll_sh = display_lls - display_lls.min()
        ll_mx = ll_sh.max() + 1e-8
        for gi, (gname, ll) in enumerate(zip(GESTURE_NAMES, ll_sh)):
            blen = int(ll / ll_mx * 90)
            col  = (80,230,120) if gi == int(np.argmax(display_lls)) else (70,70,130)
            cv2.rectangle(canvas, (8, cam_y+H-200+gi*26),
                          (8+blen, cam_y+H-186+gi*26), col, -1)
            cv2.putText(canvas, gname, (105, cam_y+H-187+gi*26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    if paused:
        ov = canvas.copy()
        cv2.rectangle(ov, (game_ox,game_oy), (game_ox+GAME_W,game_oy+GAME_H), (0,0,0), -1)
        cv2.addWeighted(ov, 0.55, canvas, 0.45, 0, canvas)
        cv2.putText(canvas, "PAUSED", (game_ox+185, game_oy+GAME_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,215,255), 3)
        cv2.putText(canvas, "hold peace sign again to unpause",
                    (game_ox+80, game_oy+GAME_H//2+40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180,180,180), 1)

    if game_over:
        ov = canvas.copy()
        cv2.rectangle(ov, (game_ox,game_oy), (game_ox+GAME_W,game_oy+GAME_H), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, "GAME OVER", (game_ox+130, game_oy+GAME_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,0,255), 3)
        cv2.putText(canvas, f"score: {score}    press R to restart",
                    (game_ox+110, game_oy+GAME_H//2+45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    if win:
        ov = canvas.copy()
        cv2.rectangle(ov, (game_ox,game_oy), (game_ox+GAME_W,game_oy+GAME_H), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, canvas, 0.5, 0, canvas)
        cv2.putText(canvas, "YOU WIN!", (game_ox+175, game_oy+GAME_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,100), 3)
        cv2.putText(canvas, f"score: {score}    press R to restart",
                    (game_ox+110, game_oy+GAME_H//2+48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

    fps_times.append(time.time() - t0)
    fps = 1.0 / (np.mean(fps_times) + 1e-8)
    cv2.putText(canvas, f"FPS:{fps:.0f}", (CANVAS_W-65,18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1)
    cv2.putText(canvas,
                "up=thumbsup  right=point  left=point  down=thumbsdown  peace(hold)=pause  palm=reset  q=quit",
                (5, CANVAS_H-4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (70,70,70), 1)

    cv2.imshow("CS5100 Capstone - HMM Pac-Man", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') and (game_over or win):
        score     = 0
        lives     = 3
        game_over = False
        win       = False
        paused    = False
        dot_grid  = fresh_dots()
        reset_positions()

landmarker.close()
cap.release()
cv2.destroyAllWindows()
