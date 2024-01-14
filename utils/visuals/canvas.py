import torch
import json
import pygame
import numpy as np
import random, os
import matplotlib.pyplot as plt
from pygame.color import THECOLORS
from torchvision import transforms
from PIL import Image
from string import ascii_letters
from typing import Optional
from utils import log
from utils.exceptions import InvalidModelException


class Canvas:

    def __init__(
            self,
            model_name: str,
            title: Optional[str] = "Canvas Window",
    ) -> None:

        if not os.path.isfile(f"saved_models/{model_name}.pt"):
            raise InvalidModelException(f"Model '{model_name}' does not exist")

        if not os.path.isdir("tmp"):
            os.mkdir("tmp")

        self.active = False
        self.title = title

        self.model = torch.load(f"saved_models/{model_name}.pt")
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def draw(self, canvas, mouse_pos, color=THECOLORS["black"]):
        """
        Draws to the specified canvas and coordinates.

        :param canvas:
        :param color:
        :param mouse_pos: 
        """
        x, y = mouse_pos
        pygame.draw.rect(canvas, color, pygame.Rect(x, y, 40, 40))

    def display(self):

        pygame.init()

        size = width, height = 500, 500

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

                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                    """ Enter """

                    if self.can_draw:
                        self.can_draw = False
                        log.info("Capturing canvas window.")

                        file_name = f"tmp-{Canvas.random_string(random.randint(4, 6))}"
                        dir_name = f"tmp/{file_name}.png"
                        pygame.image.save(CanvasWindow, dir_name)
                        self.predict_image(img_dir=dir_name)

                        CanvasWindow.fill(THECOLORS["white"])

                        log.info("Drawing can now be enabled.")
                        self.can_draw = True

                elif pygame.mouse.get_pressed()[0]:
                    """ Left Click """
                    if self.can_draw:
                        mouse_pos = pygame.mouse.get_pos()
                        self.draw(canvas=CanvasWindow, mouse_pos=mouse_pos)

                elif pygame.mouse.get_pressed()[2]:
                    """ Right Click """

                    if self.can_draw:
                        mouse_pos = pygame.mouse.get_pos()
                        self.draw(canvas=CanvasWindow, mouse_pos=mouse_pos, color=THECOLORS["white"])

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

    def _preprocess_image(self, img_dir):
        with Image.open(img_dir) as tmp_image:
            resized_img = tmp_image.resize((28, 28))
            grayscale_img = resized_img.convert("L")

        new_img = np.array(grayscale_img)
        return new_img

    def model_predict(self, processed_image) -> int:
        image_tensor = self.transforms(processed_image).to(self.device)
        image_tensor = image_tensor.unsqueeze(0)

        with torch.inference_mode():
            pred_logits = self.model(image_tensor)

        prediction = pred_logits.argmax(dim=1).item()

        return prediction

    def predict_image(self, img_dir):

        processed_image = self._preprocess_image(img_dir)
        pred = self.model_predict(processed_image=processed_image)
        os.remove(img_dir)

        log.info(f"I predict the number drawn is: {pred}", prefix="Model")




