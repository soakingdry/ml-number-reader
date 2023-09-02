# [WIP]

import pygame
import random
import cv2,os
import  time
import matplotlib.pyplot as plt

import numpy as np

from pygame.color import THECOLORS
from string import ascii_letters
from PIL import Image

from typing import Optional,Tuple
from utils import log
from utils.model.test_model import ModelTesting

class Canvas:

    def __init__(
        self, 
        title: Optional[str]="Canvas Window"
    ) -> None:

        self.active = False
        self.title = title
        self.model = ModelTesting()
    

    def draw(self, canvas, mouse_pos):
        """
        Draws to the specified canvas and coordinates.

        :param canvas: 
        :param mouse_pos: 
        """
        x,y= mouse_pos
        pygame.draw.rect(canvas, THECOLORS["black"], pygame.Rect(x,y,20,20))

    def display(self):
        
        pygame.init()

        size = width,height = 560,560

        CanvasWindow = pygame.display.set_mode(size)
        CanvasWindow.fill(THECOLORS["white"])
        self.CanvasWindow = CanvasWindow

        CanvasIcon = pygame.image.load("utils/visuals/icon.png")

        pygame.display.set_icon(CanvasIcon)
        pygame.display.set_caption(self.title)
        pygame.display.update()

        self.active = True
        self.can_draw = True

        while self.active:
            for event in pygame.event.get():
            
                if event.type == pygame.QUIT:
                    self.active = False

                elif pygame.mouse.get_pressed()[0]: 
                    """ Left Click """
                    if self.can_draw:
                        mouse_pos = pygame.mouse.get_pos()
                        self.draw(CanvasWindow, mouse_pos)

                elif pygame.mouse.get_pressed()[2]:
                    if self.can_draw:
                        self.can_draw = False
                        log.info("Capturing canvas window.")
                        file_name = f"tmp-{Canvas.random_string(random.randint(4,6))}"
                        pygame.image.save(CanvasWindow,f"tmp/{file_name}.png")
                        self.predict_image(f"tmp/{file_name}.png")
                        CanvasWindow.fill(THECOLORS["white"])
                        time.sleep(1)
                        log.info("Drawing is now enabled.")
                        self.can_draw = True

                
            pygame.display.flip()

        
        pygame.quit()

    @staticmethod
    def random_string(count: int) -> str:
        """ 
        Generates a random string which is [count] letters long

        :param int count:
        :return str:

        """

        return "".join(random.choice(ascii_letters) for x in range(count))


    def predict_image(self, img_dir):
        tmp_image = Image.open(img_dir)
        resized_img = tmp_image.resize((28,28))
        new_dir = f"tmp/res-{self.random_string(random.randint(4,6))}.png"

        resized_img.save(new_dir)
        os.remove(img_dir)

        new_img = cv2.imread(new_dir)[:,:,0]
        new_img = np.invert(np.array([new_img]))

        log.info(f"I predict this number is  a {self.model.predict(new_img)}",prefix="AI")
        plt.imshow(new_img[0], cmap=plt.cm.binary)
        plt.show()
        #os.remove(new_dir)