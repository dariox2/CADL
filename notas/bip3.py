import pygame
pygame.init()

pygame.mixer.music.load("/usr/share/sounds/purple/login.wav")
pygame.mixer.music.play()
pygame.event.wait()

