#!/usr/bin/env python3
#coding=utf-8

# checkport.py 2016-08-14

import socket
import sys

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
        ##return "BAD HOST NAME"
        pass

    if (host_addr==""):
        return "HOST NOT FOUND ["+host+"]"

    if (captive_dns_addr == host_addr):
        return "CAPTIVE DNS ["+host_addr+"]"

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #sock.settimeout(5) # ojo, transforma errno 110 en 11 (try again)
        result = sock.connect_ex((host_addr, int(port)))
        if result == 0:
            return "SOCKET OPEN ["+host_addr+":"+port+"]"
        elif result==110:
            return "TIMEOUT ["+host_addr+":"+port+"]"
        elif result==111:
            return "CONNECTION REFUSED ["+host_addr+":"+port+"]"
        else:
            return "SOCKET ERROR="+str(result)+" ["+host_addr+":"+port+"]"
    except:
        
        return "SOCKET EXCEPTION: "+sys.exc_info()[0]+" ["+host_addr+":"+port+"]"

    return "dummy"


#
# main
#

hst=sys.argv[1]
prt=sys.argv[2]

print("Host: ", hst, " port: ", prt)

r=DoesServiceExist(hst, prt)

print("Result: ", r)

# eop


