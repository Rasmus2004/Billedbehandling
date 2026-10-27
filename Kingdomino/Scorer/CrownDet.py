import os
import cv2
import numpy as np

"""
Finds crowns in an image of KingDomino tiles using template matching.
Each tile-type template is loaded and rotated to all 4 orientations.

find_crowns() returns:
    count      - total number of crowns found (after removing duplicate hits)
    annotated  - the original image with a box drawn around every crown
    detections - list of (x, y, w, h, score, tile_type, rotation_deg)
"""

user = "Rasmus_Linux"
if user == "Rasmus":
    BASE = '/Users/rasmuslykke/Desktop/Billedbehandling/Kingdomino'
elif user == "Sean":
    BASE = '/home/sean//Documents/University/3. Semester/Billedbehandling-Robot Sansning/Billedbehandling/Kingdomino'
elif user == "Rasmus_Linux":
    BASE = '/home/rasmus-lykke-sorensen/Desktop/Billedbehandling/Kingdomino'
DATA_DIR = os.path.join(BASE, "KingDdata")
TEMPLATE_DIR = os.path.join(DATA_DIR, "Templates")
BOARD_DIR = os.path.join(DATA_DIR, "CroppedBoards")

TILE_TYPES = ['wheatField', 'forest', 'lake', 'grassland', 'swamp', 'mine', 'wheatField_2', 'forest_2', 'lake_2', 'grassland_2', 'swamp_2', 'mine_2']


def load_templates():
    """Returns {tile_type: [rot0, rot90, rot180, rot270]}"""
    templates = {}
    for name in TILE_TYPES:
        img = cv2.imread(f'{TEMPLATE_DIR}/{name}.png')
        if img is None:
            raise FileNotFoundError(f'templates/{name}.png')
        rotations = [img]
        for _ in range(3):
            rotations.append(cv2.rotate(rotations[-1], cv2.ROTATE_90_CLOCKWISE))
        templates[name] = rotations
    return templates

def find_crowns(image_path, templates, threshold=0.8):
    img = cv2.imread(image_path)
    img_color = img.copy() #Copy of original image. Used for drawings etc.
    if img is None:
        raise FileNotFoundError(image_path)

    best = np.zeros(img.shape[:2], dtype=np.float32) #Holds the best match score at every pixel across all templates/rotations
    for tile_types, rotations in templates.items():
        for angle_index, template in enumerate(rotations):
            h, w = template.shape[:2]
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            rh, rw = res.shape
            best[:rh, :rw] = np.maximum(best[:rh, :rw], res) #Keeps the highest score seen so far at each pixel

    mask = (best >= threshold).astype(np.uint8)             #Binary mask of every pixel that counts as a hit
    #binary_mask_before_dilation = (mask > 0).astype(np.uint8) * 255 #Convert to binary mask for displaying before dilation

    structuring_element = np.ones((h // 2, w // 2), np.uint8)  #Structuring element for dilation. Size is half the template size to ensure that neighboring crowns arent merged together as they sit about 1 template apart.
    mask = cv2.dilate(mask, structuring_element) #Dilation merges neighbouring hits into one blob per crown
    #binary_mask_after_dilation = (mask > 0).astype(np.uint8) * 255 #Convert to binary mask for displaying
    num_labels, labels = cv2.connectedComponents(mask)      #Each blob gets its own label (0 is background)

    detections = []
    for n in range(1, num_labels):
        blob_scores = np.where(labels == n, best, 0)                    #Scores inside this blob only
        y, x = np.unravel_index(np.argmax(blob_scores), blob_scores.shape) #Position of the best hit in the blob
        detections.append((int(n), int(x), int(y), int(w), int(h), float(blob_scores[y, x]))) # Appends the detection to the list
        cv2.rectangle(img_color, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)    #Draws a rectangle around the detected crown in red color with thickness 2
        cv2.putText(img_color, str(n), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) #Puts a number above the rectangle to indicate the order of detection

    count = len(detections) #Counts the number of blobs (crowns)
    cv2.putText(img_color, f'Crowns: {count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #Writes the total number of crowns found.

    return count, img_color, detections


if __name__ == '__main__':
    templates = load_templates()
    count, annotated, detections = find_crowns(f'{BOARD_DIR}/67.jpg', templates, threshold=0.89)

    print(f'Crowns found: {count}')
    for d in detections:
        print(d)
    cv2.imshow('Crowns', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()