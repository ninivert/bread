CLI example
===========

The CLI exposes the four implemented algorithms :

::

	$ python -m bread.cli --help
	usage: bread.cli [-h] --output-file OUTPUT_FILE
	                 {budneck,expspeed,mintheta,mindist} ...

	Lineage guesser for budding yeast cells.

	positional arguments:
	  {budneck,expspeed,mintheta,mindist}
	    budneck             Guess lineage relations by looking at the budneck
	                        marker intensity along the contour of the bud.
	    expspeed            Guess lineage relations by maximizing the expansion
	                        velocity of the bud with respect to the candidate
	                        parent.
	    mintheta            Guess lineage relations by minimizing the angle
	                        between the major axis of the candidates and
	                        candidate-to-bud vector.
	    mindist             Guess lineage relations by finding the cell closest to
	                        the bud, when it appears on the segmentation.

	optional arguments:
	  -h, --help            show this help message and exit
	  --output-file OUTPUT_FILE
	                        Path to output file (csv format) (default: None)


Each algorithm is documented :

::

	python -m bread.cli budneck --help
	usage: bread.cli budneck [-h] --budneck-file BUDNECK_FILE --segmentation-file
	                         SEGMENTATION_FILE [--nn_threshold NN_THRESHOLD]
	                         [--flexible_nn_threshold FLEXIBLE_NN_THRESHOLD]
	                         [--num_frames_refractory NUM_FRAMES_REFRACTORY]
	                         [--num_frames NUM_FRAMES]
	                         [--offset_frames OFFSET_FRAMES] [--kernel_N KERNEL_N]
	                         [--kernel_sigma KERNEL_SIGMA]

	optional arguments:
	  -h, --help            show this help message and exit
	  --budneck-file BUDNECK_FILE
	                        Path to file containing budneck marker. Supported file
	                        types : tiff (as exported by Fiji) (default: None)
	  --segmentation-file SEGMENTATION_FILE
	                        Path to segmentation file. Supported file types : hd5
	                        (as exported by YeaZ) (default: None)
	  --nn_threshold NN_THRESHOLD
	                        Cell masks separated by less than this threshold are
	                        considered neighbours (default: 8)
	  --flexible_nn_threshold FLEXIBLE_NN_THRESHOLD
	                        If no nearest neighbours are found within the given
	                        threshold, try to find the closest one (default:
	                        False)
	  --num_frames_refractory NUM_FRAMES_REFRACTORY
	                        After a parent cell has budded, exclude it from the
	                        parent pool in the next frames. It is recommended to
	                        set it to a low estimate, as high values will cause
	                        mistakes to propagate in time. A value of 0
	                        corresponds to no refractory period. (default: 0)
	  --num_frames NUM_FRAMES
	                        Number of frames to make guesses for after the bud has
	                        appeared. The algorithm makes a guess for each frame,
	                        then predicts a parent by majority-vote policy
	                        (default: 5)
	  --offset_frames OFFSET_FRAMES
	                        Wait this number of frames after bud appears before
	                        guessing parent (default: 0)
	  --kernel_N KERNEL_N   Size of the gaussian smoothing kernel in pixels.
	                        larger means smoother intensity curves (default: 30)
	  --kernel_sigma KERNEL_SIGMA
	                        Number of standard deviations to consider for the
	                        smoothing kernel (default: 1)


Example of running the budneck algorithm :

::

	$ python -m bread.cli --output-file 'colony_001 budneck.csv' budneck --budneck-file data/colony001_GFP.tif --segmentation-file data/colony001_segmentation.h5 
	INFO:bread.cli:Loading segmentation...
	INFO:bread.cli:Loaded segmentation Segmentation(num_frames=181, frame_height=650, frame_width=650)
	INFO:bread.cli:Loading budneck channel movie...
	INFO:bread.cli:Loaded budneck channel movie Microscopy(num_frames=181, frame_height=650, frame_width=650)
	INFO:bread.cli:Loading guesser...
	INFO:bread.cli:Loaded guesser LineageGuesserBudLum(budneck_img=Microscopy(num_frames=181, frame_height=650, frame_width=650), segmentation=Segmentation(num_frames=181, frame_height=650, frame_width=650), nn_threshold=8, flexible_nn_threshold=False, num_frames_refractory=0, num_frames=5, offset_frames=0, kernel_N=30, kernel_sigma=1)
	INFO:bread.cli:Running guesser...
	bread/algo/lineage/_lineage.py:282: NotEnoughFramesWarning: Not enough frames in the movie (bud #80 at frame #178), requested 5, but only 3 remaining.
	bread/algo/lineage/_lineage.py:282: NotEnoughFramesWarning: Not enough frames in the movie (bud #81 at frame #178), requested 5, but only 3 remaining.
	bread/algo/lineage/_lineage.py:282: NotEnoughFramesWarning: Not enough frames in the movie (bud #82 at frame #179), requested 5, but only 2 remaining.
	INFO:bread.cli:Saving lineage...

	$ head colony_001\ budneck.csv
	# parent_id,bud_id,time_index
	-1,1,0
	-1,2,0
	2,3,4
	1,4,7
	4,5,27
	1,6,27
	2,7,28
	3,8,35
	4,9,51
