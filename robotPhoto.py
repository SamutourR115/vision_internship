import cv2
import numpy as np
import time
import os  # Bibliothèque de l'os
import sys

sys.path.append('../galatae-api/')
from robot import Robot

# Variables globales
vitesse = 70
nb_cycles_max = 1  # Nouveau paramètre pour limiter le nombre de cycles
posReplis = (300, 0, 300, 90)


def load_calibration_data():
    """Charge les paramètres de calibration de la caméra."""
    if os.path.exists('assets//calibration_camera_HBV-W202012HD.npz'):
        calibration_params = np.load('calibration_camera_HBV-W202012HD.npz', allow_pickle=True)
        mtx = np.array(calibration_params['mtx'])
        dist = np.array(calibration_params['dist'])
        print("Paramètres de calibration chargés avec succès")
        return mtx, dist
    else:
        print("Aucun paramètre de calibration trouvé")
        return None, None


def capture_images(output_folder="images", image_prefix="VisM3_6_", image_format="jpg"):
    """Capture des images en utilisant la caméra sur le robot."""

    # Création du dossier de stockage des images
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Dossier '{output_folder}' créé.")

    # Initialisation du robot
    r = Robot('/dev/ttyACM0')
    time.sleep(3)
    r.set_joint_speed(vitesse)
    r.reset_pos()

    # Chargement des paramètres de calibration
    mtx, dist = load_calibration_data()
    calibration_available = mtx is not None and dist is not None

    if calibration_available:
        print("Calibration chargée avec succès")
    else:
        print("Erreur: Calibration non chargée")
        exit()

    # Initialisation de la caméra
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la caméra.")
        return

    # Compter le nombre d'images déjà présentes dans le dossier
    existing_images = len([f for f in os.listdir(output_folder) if f.lower().endswith(f'.{image_format.lower()}')])
    img_counter = existing_images

    # Précalculer la matrice de calibration
    h, w = 720, 1280  # Taille de la caméra
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Position d'attente du robot avant la capture
    r.go_to_point(*posReplis)

    # Compteur de cycles
    nb_cycles = 0

    # Création de la fenêtre d'affichage
    cv2.namedWindow("Capture d'images", cv2.WINDOW_NORMAL)

    while nb_cycles < nb_cycles_max:
        positions = [
            (300, 0, 300, 180),
            (300, 0, 300, 180, 45),
            (300, 0, 300, 180, -45),
            (300, 15, 300, 180, 45),
            (300, 15, 300, 180, -45),
            (300, 0, 150, 180),
            (300, 0, 150, 180, 45),
            (300, 0, 150, 180, -45),
            (300, 15, 150, 180, 45),
            (300, 15, 150, 180, -45),
            (300, 0, 50, 180),
            (300, 0, 50, 180, 45),
            (300, 0, 50, 180, -45),
            (300, 15, 50, 180, 45),
            (300, 15, 50, 180, -45)
        ]

        for pos in positions:
            # Déplacer le robot à la position désirée
            r.go_to_point(*pos)
            # Attendre que le robot atteigne sa position
            time.sleep(0.5)

            # Capturer l'image après que le robot soit en position
            ret, frame = cap.read()
            if not ret:
                print("Erreur: Impossible de lire une frame de la caméra.")
                continue

            # Correction de la distorsion
            frame_undistorted = cv2.undistort(frame, mtx, dist, None, newcameramtx)
            x, y, w, h = roi
            frame_undistorted = frame_undistorted[y:y + h, x:x + w]
            frame_undistorted = cv2.resize(frame_undistorted, (frame.shape[1], frame.shape[0]))

            # Rotation de l'image
            flipFrameUndist = cv2.rotate(frame_undistorted, cv2.ROTATE_180)

            # Afficher l'image
            cv2.imshow("Capture d'images", flipFrameUndist)
            cv2.waitKey(1)  # Rafraîchir l'affichage

            # Enregistrer l'image
            img_name = f"{image_prefix}{img_counter + 1}.{image_format}"
            img_path = os.path.join(output_folder, img_name)
            cv2.imwrite(img_path, flipFrameUndist)
            img_counter += 1
            print(f"Image sauvegardée: {img_path}")

        # Attendre un moment après avoir terminé toutes les positions
        time.sleep(5)

        # Retourner à la position de repli
        r.go_to_point(*posReplis)
        r.go_to_foetus_pos()

        nb_cycles += 1

        # Vérifier si l'utilisateur veut arrêter
        key = cv2.waitKey(100)
        if key == 27:  # ESC key
            break

    # Libération des ressources
    cap.release()
    cv2.destroyAllWindows()
    print(f"{img_counter - existing_images} nouvelles images ont été sauvegardées dans le dossier '{output_folder}'.")
    print(f"Nombre total d'images dans le dossier: {img_counter}")


if __name__ == "__main__":
    capture_images()