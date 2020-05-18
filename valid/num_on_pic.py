import cv2
import random


class Draw:
    def __init__(self, original, draw, image_path):
        self.origin = original
        self.draw = draw
        self.image_path = image_path
        self.array = None
        self.draw_into_image()

    def draw_into_image(self):
        x = self.origin.shape[1]
        y = self.origin.shape[0]
        size = random.randint(50, 200)
        x = random.randint(0, x-size)
        y = random.randint(x, y)

        cv2.putText(self.origin, str(self.draw), (x, y), cv2.FONT_HERSHEY_PLAIN, 7, (50, 50, 50), 3, cv2.LINE_8)
        wrote = cv2.imwrite(self.image_path, self.origin)
        self.array = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
