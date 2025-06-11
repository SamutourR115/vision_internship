import cv2
import numpy as np
import glob
import os
import time
from datetime import datetime
import logging

# === CONFIGURATION LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === PARAMÈTRES DE LA CAMÉRA (INFORMATIONS) ===
LARGEUR_CAMERA = 1280
HAUTEUR_CAMERA = 720
MODELE_CAMERA = "HBV-W202012HD"
CAPTEUR_CAMERA = "OV9726 (1/6\")"
RESOLUTION_CAMERA = "1280 x 720"
CHAMP_VISION_CAMERA = "50°"
FOCALE_CAMERA = "3.2 mm"

# === PARAMÈTRES DE L'ÉCHIQUIER DE CALIBRATION ===
NB_COINS_X = 9  # Nombre de coins intérieurs en largeur
NB_COINS_Y = 6  # Nombre de coins intérieurs en hauteur
TAILLE_CARRE_MM = 25.0  # Taille réelle d'un carré en mm - À AJUSTER SI NÉCESSAIRE
DOSSIER_IMAGES_CALIBRATION = '/Users/samos/Documents/StageCv/GaleVision/calib_pictures/images_calibration2'
NOM_FICHIER_CALIBRATION = 'calibration_camera_HBV-W202012HD.2.npz'
MIN_IMAGES_CALIBRATION = 5

def charger_images_calibration(dossier_images, nb_coins_x, nb_coins_y, taille_carre_mm):
    """
    Charge les images de calibration et extrait les points de l'échiquier.
    """
    logger.info(f"Chargement des images de calibration depuis {dossier_images}...")

    if not os.path.exists(dossier_images):
        logger.error(f"Le dossier {dossier_images} n'existe pas.")
        return None, None, None

    points_objet = np.zeros((nb_coins_x * nb_coins_y, 3), np.float32)
    points_objet[:, :2] = np.mgrid[0:nb_coins_x, 0:nb_coins_y].T.reshape(-1, 2) * taille_carre_mm

    points_objets = []
    points_images = []
    images_trouvees = []

    images = glob.glob(os.path.join(dossier_images, '*.jpg'))
    if not images:
        logger.error(f"Aucune image trouvée dans {dossier_images}.")
        return None, None, None

    logger.info(f"Trouvé {len(images)} images de calibration.")

    for i, image_path in enumerate(images):
        logger.info(f"Traitement de l'image {i+1}/{len(images)}: {os.path.basename(image_path)}")
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Impossible de lire l'image {image_path}.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, (nb_coins_x, nb_coins_y), None)

        if ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            points_objets.append(points_objet)
            points_images.append(corners2)
            images_trouvees.append(img)
            logger.info(f"Échiquier détecté dans l'image {i+1}.")
        else:
            logger.warning(f"Échiquier non détecté dans l'image {i+1}.")

    if len(points_objets) < MIN_IMAGES_CALIBRATION:
        logger.error(f"Pas assez d'images avec échiquier détecté (minimum {MIN_IMAGES_CALIBRATION}).")
        return None, None, None

    return points_objets, points_images, images_trouvees[0]

def calibrer_camera(utiliser_images_existantes=False):
    logger.info("=== Programme de calibration pour caméra HBV-W202012HD ===")
    logger.info(f"Modèle: {MODELE_CAMERA}, Capteur: {CAPTEUR_CAMERA}, Résolution: {RESOLUTION_CAMERA}, Champ de vision: {CHAMP_VISION_CAMERA}, Focale: {FOCALE_CAMERA}")
    logger.info(f"Taille d'un carré de l'échiquier: {TAILLE_CARRE_MM} mm")
    logger.info("Vérifiez que cette valeur est correcte.")

    if not os.path.exists(DOSSIER_IMAGES_CALIBRATION):
        os.makedirs(DOSSIER_IMAGES_CALIBRATION)
        logger.info(f"Dossier '{DOSSIER_IMAGES_CALIBRATION}' créé.")

    points_objets = []
    points_images = []
    gray_shape = (HAUTEUR_CAMERA, LARGEUR_CAMERA)  # Initialisation par défaut

    if utiliser_images_existantes:
        points_objets, points_images, img = charger_images_calibration(
            DOSSIER_IMAGES_CALIBRATION, NB_COINS_X, NB_COINS_Y, TAILLE_CARRE_MM)
        if points_objets is None:
            logger.info("Impossible de charger les images existantes, passage en mode capture.")
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_shape = gray.shape[::-1]
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                points_objets, points_images, gray_shape, None, None)
            erreur_moyenne = calculer_erreur_reprojection(points_objets, points_images, rvecs, tvecs, mtx, dist)
            logger.info(f"Erreur de re-projection moyenne avec les images existantes: {erreur_moyenne:.4f}")
            distance_manuelle = float(input("\nEntrez manuellement la distance caméra-plan de travail (mm): "))
            sauvegarder_calibration(mtx, dist, rvecs, tvecs, distance_manuelle)
            return

    # Mode capture de nouvelles images
    logger.info("\n=== Mode de capture de nouvelles images de calibration ===")
    logger.info("Instructions:")
    logger.info("1. Placez un échiquier de calibration devant la caméra.")
    logger.info("2. Appuyez sur 'c' pour capturer une image quand l'échiquier est détecté.")
    logger.info("3. Capturez au moins {} images de l'échiquier dans différentes positions.".format(MIN_IMAGES_CALIBRATION))
    logger.info("4. Appuyez sur 'q' pour quitter le mode capture et lancer la calibration.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Impossible d'ouvrir la caméra.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGEUR_CAMERA)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HAUTEUR_CAMERA)

    nb_images_capturees = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Erreur lors de la capture de l'image.")
            break

        affichage = frame.copy()
        cv2.putText(affichage, f"Images capturées: {nb_images_capturees}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(affichage, "Appuyez sur 'c' pour capturer, 'q' pour quitter",
                    (10, HAUTEUR_CAMERA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, (NB_COINS_X, NB_COINS_Y), None)

        if ret_corners:
            cv2.putText(affichage, "Échiquier détecté", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.drawChessboardCorners(affichage, (NB_COINS_X, NB_COINS_Y), corners, ret_corners)

        cv2.imshow('Calibration Camera', affichage)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c') and ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            points_objets.append(np.zeros((NB_COINS_X * NB_COINS_Y, 3), np.float32))
            points_objets[-1][:, :2] = np.mgrid[0:NB_COINS_X, 0:NB_COINS_Y].T.reshape(-1, 2) * TAILLE_CARRE_MM
            points_images.append(corners2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nom_fichier = os.path.join(DOSSIER_IMAGES_CALIBRATION, f'calibration_{timestamp}.jpg')
            cv2.imwrite(nom_fichier, frame)
            nb_images_capturees += 1
            logger.info(f"Image {nb_images_capturees} capturée.")
            time.sleep(0.5)

        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(points_objets) < MIN_IMAGES_CALIBRATION:
        logger.error(f"Pas assez d'images capturées pour une calibration fiable (minimum {MIN_IMAGES_CALIBRATION}).")
        return

    logger.info(f"\nCalibration en cours avec {len(points_objets)} images...")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Utiliser la dernière frame pour la taille
    gray_shape = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        points_objets, points_images, gray_shape, None, None)

    erreur_moyenne = calculer_erreur_reprojection(points_objets, points_images, rvecs, tvecs, mtx, dist)
    logger.info(f"Erreur de re-projection moyenne: {erreur_moyenne:.4f}")

    distance_manuelle = float(input("\nEntrez manuellement la distance caméra-plan de travail (mm): "))
    sauvegarder_calibration(mtx, dist, rvecs, tvecs, distance_manuelle)

def calculer_erreur_reprojection(points_objets, points_images, rvecs, tvecs, mtx, dist):
    """Calcule l'erreur de re-projection moyenne."""
    total_erreur = 0
    for i in range(len(points_objets)):
        imgpoints2, _ = cv2.projectPoints(points_objets[i], rvecs[i], tvecs[i], mtx, dist)
        erreur = cv2.norm(points_images[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_erreur += erreur
    return total_erreur / len(points_objets)

def sauvegarder_calibration(mtx, dist, rvecs, tvecs, camera_to_workspace_distance):
    """Sauvegarde les paramètres de calibration dans un fichier .npz."""
    camera_to_workspace = np.eye(4)
    camera_to_workspace[2, 3] = camera_to_workspace_distance
    np.savez(NOM_FICHIER_CALIBRATION,
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs,
             camera_to_workspace=camera_to_workspace)
    logger.info(f"Paramètres de calibration sauvegardés dans '{NOM_FICHIER_CALIBRATION}'.")
    logger.info("\nParamètres de calibration:")
    logger.info(f"Matrice de la caméra (mtx):\n{mtx}")
    logger.info(f"\nCoefficients de distorsion (dist):\n{dist}")
    logger.info(f"\nDistance caméra-plan de travail (translation Z):\n{camera_to_workspace_distance:.2f} mm")
    logger.info("\nPour utiliser la calibration dans vos applications:")
    logger.info(f"1. Chargez les paramètres avec: data = np.load('{NOM_FICHIER_CALIBRATION}')")
    logger.info("2. Extrayez les paramètres: mtx = data['mtx'], dist = data['dist'], camera_to_workspace = data['camera_to_workspace']")
    logger.info("3. Utilisez cv2.undistort(image, mtx, dist) pour corriger la distorsion d'image.")

if __name__ == "__main__":
    logger.info("=== Programme de calibration de caméra ===")
    logger.info("Choisissez le mode de calibration :")
    logger.info("1. Utiliser des images de calibration existantes")
    logger.info("2. Prendre de nouvelles images de calibration")
    logger.info("3. Quitter")

    while True:
        choix = input("Votre choix (1, 2 ou 3): ")

        if choix == "1":
            calibrer_camera(utiliser_images_existantes=True)
            break  # Sortir de la boucle après la calibration
        elif choix == "2":
            calibrer_camera(utiliser_images_existantes=False)
            break  # Sortir de la boucle après la calibration
        elif choix == "3":
            logger.info("Programme de calibration terminé.")
            break  # Quitter le programme
        else:
            logger.warning("Choix invalide. Veuillez entrer 1, 2 ou 3.")