#!/usr/bin/env python3
#coding=utf-8

# checkport.py 2016-08-14

import socket
import sys
import pyglet
import time

def DoesServiceExist(host, port):
    captive_dns_addr = ""
    host_addr = ""

    try:
        captive_dns_addr = socket.gethostbyname("dummyinexistente12345.com")
    except:
        pass

    try:
        host_addr = socket.gethostbyname(host)
    except:
        pass

    if (host_addr==""):
        return -1

    if (captive_dns_addr == host_addr):
        return -2

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3) # ojo, transforma errno 110 en 11 (try again)
        result = sock.connect_ex((host_addr, int(port)))
        return result
    except:
        return -3

    return -999


#
# main
#

hst=sys.argv[1]
prt=sys.argv[2]

music = pyglet.media.load('/usr/share/sounds/purple/login.wav', streaming=False)

while (True):
  r=DoesServiceExist(hst, prt)
  print(r)
  if (r!=0):
    music.play()
  time.sleep(3)


# eop


