import cv2
import numpy as np
import os

"""
Finds crowns in an image of KingDomino tiles using template matching.
Each tile-type template is loaded and rotated to all 4 orientations.

find_crowns() returns:
    count      - total number of crowns found (after removing duplicate hits)
    annotated  - the original image with a box drawn around every crown
    heatmap    - colour map of the best match score at every pixel (for tuning the threshold)
    detections - list of (x, y, w, h, score, tile_type, rotation_deg)
"""
user = "Rasmus"
if user == "Rasmus":
    BASE = '/Users/rasmuslykke/Desktop/Billedbehandling/Kingdomino'
elif user == "Sean":
    BASE = 'seans path'
DATA_DIR = os.path.join(BASE, "KingDdata")
TEMPLATE_DIR = os.path.join(DATA_DIR, "templates")
BOARD_DIR = os.path.join(DATA_DIR, "CroppedBoards")

TILE_TYPES = ['wheatField', 'forest', 'lake', 'grassland', 'swamp', 'mine']


def load_templates():
    """Returns {tile_type: [rot0, rot90, rot180, rot270]}"""
    templates = {} #Dictionary to hold the templates for each tile type
    for name in TILE_TYPES: #Iterate over the tile types and load the corresponding template image
        img = cv2.imread(f'{TEMPLATE_DIR}/{name}.png') #Load the template image for the current tile type
        if img is None:
            raise FileNotFoundError(f'templates/{name}.png')
        rotations = [img] #List to hold the four rotations of the template
        for _ in range(3):
            rotations.append(cv2.rotate(rotations[-1], cv2.ROTATE_90_CLOCKWISE)) #Rotate the last image in the list by 90 degrees clockwise and append it to the rotations list
        templates[name] = rotations #Store the four rotations of the template in the templates dictionary
    return templates

"""
Non-max supression was made by Claude to remove overlapping boxes issues in template matching.
The group understands the core principle but the idea of using NMS was suggested and implemented by Claude.
"""
def non_max_suppression(boxes, scores, max_overlap=0.3):
    """
    Greedy NMS. boxes is an (N, 4) array of x, y, w, h.
    Keeps the highest-scoring box and drops any box overlapping it by more than
    max_overlap (IoU). Returns the indices of the boxes to keep.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    order = np.argsort(scores)[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]

        ix1 = np.maximum(x1[i], x1[rest])
        iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest])
        iy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou = inter / (areas[i] + areas[rest] - inter)

        order = rest[iou <= max_overlap]
    return keep


def find_crowns(image_path, templates, threshold=0.8, max_overlap=0.3):
    img = cv2.imread(image_path)
    img_color = img.copy() #Copy of original image. Used for drawings etc.
    if img is None:
        raise FileNotFoundError(image_path)

    boxes, scores = [], [], #Empty lists to hold the boxes and scores of the template matches
    for tile_types, rotations in templates.items():
        for angle_index, template in enumerate(rotations):
            h, w = template.shape[:2] #Get the height and width of the template
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED) #Perform template matching using normalized cross-correlation

            ys, xs = np.where(res >= threshold) #Find the coordinates of the pixels where the match score is above the threshold
            for x, y in zip(xs, ys): #Iterate over the coordinates of the matching pixels
                boxes.append((x, y, w, h)) #Append the coordinates and dimensions of the matching box to the boxes list
                scores.append(res[y, x]) #Append the match score of the matching box to the scores list

    boxes = np.array(boxes).reshape(-1, 4)  #Convert to numpy array and reshape to (N, 4) array
    scores = np.array(scores)               #Converts scores to numpy array
    keep = non_max_suppression(boxes, scores, max_overlap) #Applies NMS to remove overlapping boxes and multiple detections

    detections = []
    for n, i in enumerate(keep, start=1):
        x, y, w, h = boxes[i]                                                                   # Get the coordinates and dimensions of the kept box
        detections.append((int(x), int(y), int(w), int(h), float(scores[i])))                   # Appends the detection to the list
        cv2.rectangle(img_color, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)    #Draws a rectangle around the detected crown in red color with thickness 2
        cv2.putText(img_color, str(n), (int(x), int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) #Puts a number above the rectangle to indicate the order of detection

    count = len(detections) #Counts the number of detections post NMS
    cv2.putText(img_color, f'Crowns: {count}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) #Writes the total number of crowns found.

    return count, img_color, detections


if __name__ == '__main__':
    templates = load_templates()
    count, annotated, detections = find_crowns(f'{BOARD_DIR}/2.jpg', templates, threshold=0.68, max_overlap=0.3)

    print(f'Crowns found: {count}')
    for d in detections:
        print(d)
    
    cv2.imshow('Crowns', annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()