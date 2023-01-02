
# Computer Vision Assignment

## Project description

A museum is developing a virtual reality (VR) reconstruction of a region of London, showing how it appeared in about 1900. They want visitors to be able to place a pointer on a map of London and use the position and orientation of the pointer to determine the location and direction of the view in the reconstruction.

Your program must accept precisely one argument from the command line, the name of the image to be processed, as in:

'python3 mapreader.py develop/develop-001.png'

It should output two lines in the following format:

'POSITION 0.673 0.212'
'BEARING 316.4'

The two numbers following POSITION represent the location of the point of the red pointer, being respectively the distance along the bottom of the map and the distance up its side, with the origin at the bottom left-hand (south west) corner. These numbers should both be in the range 0–1. The number following BEARING should be the angle in which the red pointer is pointing, given in degrees measured clockwise from north.

Any other output it generates is ignored. Your submitted program must not display any images.

## Code structure

The code developed for the project is named mapreader.py and it can be found in this same directory.

The first part of the code defines a set of routines used throughout the assignment. The main program first segments the map (region of interest) and rectifies the image. It then identifies the orientation arrow to determine if the map is upside-down and rotates it if necessary. Finally, the pointer arrow is processed to locate its tip and calculate its bearing. These values are printed at the end.
