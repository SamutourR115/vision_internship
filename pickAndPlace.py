import cv2
import numpy as np
import time
import sys
import os
import logging
sys.path.append('../galatae-api/')
from robot import Robot
from ultralytics import YOLO
import threading
import math

# --- CONFIGURATION LOGGING ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Paramètres décalage caméra ---
END_EFFECTOR_LENGTH = 70
stop_requested = False

# --- Conversion pixel vers position robot (repère outil) ---
def convertir_pixel_en_mm(point_px, H):
    # Transformer le point en coordonnées homogènes (pixels → mm via homographie)
    point_np = np.array([[point_px]], dtype='float32')  # shape: (1, 1, 2)
    point_mm_cam = cv2.perspectiveTransform(point_np, H)[0][0]  # shape: (2,)

    # Conversion repère caméra → repère robot
    x_robot = float(point_mm_cam[0])
    y_robot = float(point_mm_cam[1])

    # Retourne une liste Python native
    return [x_robot, y_robot]

# --- Chargement de la calibration ---
def load_Calibration():
    try:
        calib_path = "/Users/samos/Documents/StageCv/GaleVision/assets/calibration_camera_HBV-W202012HD.2.npz"
        if os.path.exists(calib_path):
            calib = np.load(calib_path)
            logger.info("Matrice de calibration chargée avec succès.")
            return calib['mtx'], calib['dist'], calib['camera_to_workspace']
        else:
            logger.error("Fichier de calibration non trouvé.")
            return None, None, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la calibration : {e}")
        return None, None, None

# --- Vérifie si l'utilisateur veut arrêter via la console ---
def check_for_stop():
    global stop_requested
    while True:
        user_input = input("Tapez 'stop' pour arrêter : ")
        if user_input.strip().lower() == "stop":
            logger.info("Arrêt demandé par l'utilisateur via console.")
            stop_requested = True
            break

#--- Chargement du programme de pick and place ---
def pick_and_place():

    logger.info("Initialisation du robot")
    try:
        robot = Robot('/dev/tty.usbmodem1201')
        time.sleep(3)
        robot.set_joint_speed(30)
        robot.reset_pos()
        robot.calibrate_gripper()
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du robot : {e}")
        return

    logger.info("Chargement du modèle YOLO")
    try:
        model = YOLO('/Users/samos/Documents/StageCv/GaleVision/assets/best.pt')
    except Exception as e:
        logger.error(f"Erreur YOLO : {e}")
        return

    logger.info("Chargement calibration")
    mtx, dist, _ = load_Calibration()
    H = np.load("/Users/samos/Documents/StageCv/GaleVision/assets/homographie.npy")

    placement_points = {
        "M3*6": [275, 320, 140, 180, 0],
        "M3*8": [345, 320, 140, 180, 0],
        "M3*10": [275, 250, 140, 180, 0],
        "M3*12": [345, 250, 140, 180, 0],
        "M3*16": [315, 290, 140, 180, 0],
    }

    robot.go_to_point([300, 0, 150, 180, 0])
    time.sleep(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Erreur : caméra non disponible.")
        robot.go_to_foetus_pos()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    x_roi, y_roi, w_roi, h_roi = roi

    threading.Thread(target=check_for_stop, daemon=True).start()

    last_detection_time = time.time()

    try:
        while not stop_requested:
            #Vider le cache d'opencv
            for _ in range(5):
                cap.grab()
            ret, frame = cap.read()
            if not ret:
                logger.error("Erreur de lecture image.")
                break

            undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)
            undistorted = undistorted[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi]

            results = model(undistorted, verbose=False)[0]

            cv2.imshow("Vue corrigée", undistorted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Si des vis détectées
            if len(results.boxes) > 0:
                last_detection_time = time.time()

                detected_screws = []
                for box in results.boxes:
                    confidence = float(box.conf[0])  # Score de confiance YOLO
                    if confidence < 0.65:
                        logger.info(f"Objet détecté ignoré : confiance trop faible ({confidence:.2f})")
                        continue  # Ne pas traiter cette détection

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    pos_robot = convertir_pixel_en_mm((center_x, center_y), H)

                    dx = x2 - x1
                    dy = y2 - y1
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    angle_deg = round(max(min(angle_deg, 25), -25), 2)

                    detected_screws.append({
                        "class": class_name,
                        "position": pos_robot,
                        "angle": angle_deg
                    })

                    logger.info(
                        f"Vis '{class_name}' détectée (confiance {confidence:.2f}) en {pos_robot}° avec angle {angle_deg}°.")

                for screw in detected_screws:
                    class_name = screw["class"]
                    x_pos, y_pos = screw["position"]
                    angle = screw["angle"]

                    robot.go_to_point([x_pos, y_pos, 150, 180, 0,0])
                    robot.go_to_point([x_pos, y_pos, 150, 180, angle,0])
                    robot.go_to_point([x_pos, y_pos, 53, 180, angle,0])
                    robot.go_to_point([x_pos, y_pos, 53, 180, angle, 70])
                    robot.go_to_point([x_pos, y_pos, 150, 180, angle,70])

                    if class_name in placement_points:
                        logger.info(f"Placement de la vis {class_name}")
                        robot.go_to_point(placement_points[class_name])
                        robot.open_gripper()

                    robot.go_to_point([300, 0, 150, 180, 0, 0])
                    robot.open_gripper()
                    time.sleep(0.5)

            else:
                # Si pas de détection, vérifier timeout
                if time.time() - last_detection_time > 60:
                    logger.info("Aucune vis détectée depuis 5 secondes. Retour en position foetus.")
                    robot.go_to_foetus_pos()
                    break

                time.sleep(1)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.go_to_foetus_pos()
        logger.info("Fin de la boucle pick and place.")

if __name__ == "__main__":
    pick_and_place()
    logger.info("Program finished")