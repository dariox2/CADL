##import winsound
##Freq = 2500 # Set Frequency To 2500 Hertz
##Dur = 1000 # Set Duration To 1000 ms == 1 second
##winsound.Beep(Freq,Dur)

from playsound import playsound
# wav works on all platforms. mp3 works on OS X. Other filetype/OS combos may work but have not been tested.
playsound('/usr/share/sounds/purple/alert.wav')

