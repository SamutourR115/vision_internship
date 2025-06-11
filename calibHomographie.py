import sys
import os
import time
import cv2
import numpy as np
sys.path.append('../galatae-api/')
from robot import Robot

# --- Initialisation du robot ---
robot = Robot('/dev/tty.usbmodem1201')
time.sleep(3)
robot.set_joint_speed(30)
robot.reset_pos()

# --- Chargement de la calibration caméra ---
def load_calibration():
    path = "assets/calibration_camera_HBV-W202012HD.2.npz"
    if os.path.exists(path):
        data = np.load(path)
        return data["mtx"], data["dist"], data["camera_to_workspace"]
    else:
        print("[ERREUR] Fichier de calibration non trouvé.")
        return None, None, None

# --- Sélection d’un point à la souris ---
def selectionner_point(image):
    point = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            point.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Sélection", image)

    cv2.imshow("Sélection", image)
    cv2.setMouseCallback("Sélection", on_click)

    while len(point) == 0:
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    return point[0]

# --- Calibration par homographie ---
def calibrer_homographie():
    cap = cv2.VideoCapture(0)
    mtx, dist, _ = load_calibration()
    if mtx is None: return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    # Points connus en mm dans le repère robot (repère outil ou plan)
    points_mm = [
        (320.02, 56.88), (359.97, 56.86), (391.95, 56.87),
        (329.94, 0),     (364.93, 0),     (391.98, 0),
        (319.94,-48.89), (360.02,-48.85), (384.92,-48.79)
    ]
    points_px = []

    print("[INFO] Clique sur les points demandés.")
    robot.go_to_point([300, 0, 150, 180, 0])

    for i, (x_mm, y_mm) in enumerate(points_mm):
        ret, frame = cap.read()
        if not ret:
            print("[ERREUR] Capture échouée.")
            continue

        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
        x_roi, y_roi, w_roi, h_roi = roi
        undistorted = undistorted[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]
        display_frame = undistorted.copy()

        texte = f"Point {i+1} : {x_mm:.1f}, {y_mm:.1f} mm"
        cv2.putText(display_frame, texte, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        point = selectionner_point(display_frame)
        points_px.append(point)

    cap.release()

    # Homographie
    pts_px = np.array(points_px, dtype=np.float32)
    pts_mm = np.array(points_mm, dtype=np.float32)
    H, _ = cv2.findHomography(pts_px, pts_mm)

    np.save("/assets", H)
    print("[INFO] Homographie sauvegardée dans homographie.npy")
    return H

# --- Lancement principal ---
if __name__ == "__main__":
    calibrer_homographie()