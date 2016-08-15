import pyglet
import os

working_dir = os.path.dirname(os.path.realpath(__file__))
#pyglet.resource.path = [os.path.join(working_dir,'Images')]
pyglet.resource.path = working_dir

music = pyglet.resource.media('Pew_Pew-DKnight556-1379997159.mp3')
music.play()

pyglet.app.run()


