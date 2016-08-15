#!/usr/bin/env python3
#coding=utf-8

import pyglet
import time
import gc

music = pyglet.media.load('/usr/share/sounds/purple/login.wav', streaming=False)

music.play()
time.sleep(3)

# ...etc

music.play()
time.sleep(3)

#etc

# funca ???
del music
gc.collect()

# cuando se ejecuta desde el shell, tira:
#python3: pthread_mutex_lock.c:326: __pthread_mutex_lock_full: Assertion `robust || (oldval & 0x40000000) == 0' failed.
#Aborted

#eop



