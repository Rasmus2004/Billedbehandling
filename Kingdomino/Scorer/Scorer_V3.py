import cc as tiles                  # Imports the tile classifier and grassfire grouping (cc.py) under the name "tiles"
import KingD_V1 as crown_finder     # Imports the crown template matching (KingD_V1.py) under the name "crown_finder"
import cv2
import os

# Path to ONE cropped 500x500 board picture, both scripts read this same picture
user = "Rasmus"
if user == "Rasmus":
    BASE = '/Users/rasmuslykke/Desktop/Billedbehandling/Kingdomino'
elif user == "Sean":
    BASE = 'seans path'
DATA_DIR = os.path.join(BASE, "KingDdata")
IMAGE_PATH = os.path.join(DATA_DIR, "CroppedBoards/2.jpg")  # Change the number to test different boards, e.g. "2.jpg" or "3.jpg"
THRESHOLD = 0.68        # How good a template match must be to count as a crown (same value as in KingD_V1)
MAX_OVERLAP = 0.3       # How much two crown boxes may overlap before one is thrown away (same value as in KingD_V1)


def main():                                                                         # Runs the whole scoring for one board
    # ---------- Step 1: find the tile groups (cc.py) ----------
    blurred = tiles.mean_blur_image(IMAGE_PATH, tiles.KERNEL_SIZE)                  # Blurs the picture like cc.py does
    mosaic, values = tiles.annotate_tiles(blurred)                                  # Gets the mean HSV of each of the 25 tiles (the mosaic picture is not used)
    results = tiles.classify_all(values)                                            # Gives every tile a terrain type from its HSV values
    results = tiles.enforce_single_crown(results, values, IMAGE_PATH)               # Makes sure exactly one tile is the castle tile, like cc.py does
    groups = tiles.find_groups(results)                                             # Groups neighbouring tiles of the same type: {group_id: (type, [(row, col), ...])}

    # ---------- Step 2: find the crowns (KingD_V1.py) ----------
    templates = crown_finder.load_templates()                                       # Loads the crown templates in all 4 rotations
    count, annotated, detections = crown_finder.find_crowns(IMAGE_PATH, templates, THRESHOLD, MAX_OVERLAP)  # Finds all crowns, detections holds one entry per crown

    # ---------- Step 3: put every crown on the tile it sits on ----------
    crowns_on_tile = {}                                                             # Will hold how many crowns each tile has: {(row, col): number of crowns}
    for (x, y, w, h, score) in detections:                                          # Goes through every crown that was found
        row = (y + h // 2) // tiles.TILE_SIZE                                       # Crown centre y divided by the tile size gives the tile row
        col = (x + w // 2) // tiles.TILE_SIZE                                       # Crown centre x divided by the tile size gives the tile column
        crowns_on_tile[(row, col)] = crowns_on_tile.get((row, col), 0) + 1          # Adds 1 to the crown count of that tile (starts at 0 if the tile has none yet)

    # ---------- Step 4: score = crowns * connected tiles, for every group ----------
    total_score = 0                                                                 # The final score of the board, starts at 0
    for group_id, (name, members) in groups.items():                                # Goes through every group of connected tiles
        if tiles.is_crown(name):                                                    # Checks if the group is the initial crown tile (the castle)
            continue                                                                # The castle gives no points, so skip it
        num_connected_tiles = len(members)                                          # The number of tiles in this group
        num_crowns = sum(crowns_on_tile.get(pos, 0) for pos in members)             # Adds up the crowns on all tiles in this group (0 for tiles without crowns)
        group_score = num_crowns * num_connected_tiles                              # The score of this group
        total_score += group_score                                                  # Adds the group score to the board score
        print(f'{name}: {num_crowns} crown(s) * {num_connected_tiles} tile(s) = {group_score}')  # Prints how the group score was made

    print(f'Crowns found: {count}')                                                 # Prints the total number of crowns found, to check against the picture
    print(f'Total score: {total_score}')                                            # Prints the final score of the board
        # ---------- Step 5: show the picture with the crowns and the points ----------
    cv2.putText(annotated, f'Points: {total_score}', (10, 55),                      # Writes the total points on the picture, just below the "Crowns: N" text that find_crowns already wrote
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)                      # Same font, size, green colour and thickness as the "Crowns: N" text
    cv2.imshow('KingDomino score', annotated)                                       # Opens a window that shows the picture with the crown boxes, crown count and points
    cv2.waitKey(0)                                                                  # Keeps the window open until any key is pressed
    cv2.destroyAllWindows()                                                         # Closes the window again


if __name__ == '__main__':                                                          # True only when this file is run directly
    main()                                                                          # Runs the scoring