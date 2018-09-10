# Data
To train or evaluate on additional datasets, 
please prepare the data following the format in demo.
Tools that help prepare the format are coming soon.

This implementation assumes the GoP structue described in the
paper.
Namely, frame 1-12 forms the first GoP, frame 13-24 forms the next GoP, and so on.
The first level interpolates frame 7 using frame 1 and frame 13.
The second level interpolates frame 4 using frame 1 and 7, 
and interpolates frame 10 using frame 7 and 13.
The third level interpolates the remaining.

The motion vector/optical flow dependency should follow the structure.
For example, the motion information to interpolate frame 7 should include
the motion from frame 1 to frame 7, and the motion from frame 13 to frame 7.
One can use any existing motion estimation or optical flow algorithms to get
the motion.
Tools to facilitate motion extraction are coming soon. 
