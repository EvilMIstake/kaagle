
import os
from time import time

from pygame import display, init, Surface
from pygame import event, QUIT, MOUSEBUTTONUP, MOUSEBUTTONDOWN
from pygame import NOFRAME, HWACCEL
from pygame.transform import smoothscale

from numpy import array, matmul, int_

from isometric_field import IsoField


def resolution_handler(w: float, h: float) -> array:
    return array(
        (
            (w / 1920, 0),
            (0, h / 1080)
        )
    )


def window_handler(size: array) -> Surface:
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    screen = display.set_mode(size, HWACCEL, vsync=True)
    return screen


def events_handler(dictionary: dict) -> None:
    for event_ in event.get():
        eventType = event_.type

        dictionary[QUIT] = eventType != QUIT
        dictionary[MOUSEBUTTONUP] = eventType == MOUSEBUTTONUP
        dictionary[MOUSEBUTTONDOWN] = eventType == MOUSEBUTTONDOWN

        eventDictionary = event_.__dict__
        dictionary[1234] = eventDictionary.get('button', 0)
        dictionary[1235] = array(eventDictionary.get('pos', None))


def main():
    SIZE = array((1280, 720))
    RESOLUTION_MATRIX = resolution_handler(display.Info().current_w, display.Info().current_h)
    REAL_SCREEN_SIZE = matmul(SIZE, RESOLUTION_MATRIX).astype(int_)
    FPS = 60

    screen = window_handler(REAL_SCREEN_SIZE)
    display.set_caption('')

    currentTime = time()
    accumulator = 0
    dt = 1 / FPS
    events = {1236: array((0, 0))}

    iso_field = IsoField(30, 25)

    while events.get(QUIT, True):
        events_handler(events)

        newTime = time()
        frameTime = min(newTime - currentTime, 0.25)
        currentTime = newTime
        accumulator += frameTime

        while accumulator >= dt:
            iso_field.update()
            accumulator -= dt

        alpha = accumulator * FPS

        screen.fill((0, 0, 0, 0))
        iso_field.draw(screen)
        display.update()


if __name__ == "__main__":
    init()
    main()
