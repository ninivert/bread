import argparse
import pathlib
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('bread.cli')
import bread.algo.lineage

__all__ = ['main']

def main():
	parser_main = argparse.ArgumentParser(
		prog='bread.cli',
		description='Lineage guesser and tracker for budding yeast cells.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	subparsers_main = parser_main.add_subparsers(
		dest='action',
		required=True
	)

	# Lineage parser

	parser_lineage = subparsers_main.add_parser(
		'lineage',
		help='Guess cell lineages.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser_lineage.add_argument(
		'--segmentation-file',
		help='Path to segmentation file. Supported file types : hd5 (as exported by YeaZ)',
		type=pathlib.Path,
		required=True
	)
	parser_lineage.add_argument(
		'--output-file',
		help='Path to output file (csv format)',
		type=pathlib.Path,
		required=True
	)
	parser_lineage.add_argument(
		'--nn_threshold',
		help='Cell masks separated by less than this threshold are considered neighbours',
		default=bread.algo.lineage._lineage.LineageGuesser.nn_threshold,
		type=float,
	)
	parser_lineage.add_argument(
		'--flexible_nn_threshold',
		help='If no nearest neighbours are found within the given threshold, try to find the closest one',
		default=bread.algo.lineage._lineage.LineageGuesser.flexible_nn_threshold,
		type=bool,
	)

	subparsers_lineage = parser_lineage.add_subparsers(
		dest='lineage_algo',
		required=True,
	)
	
	parser_lineage_expspeed = subparsers_lineage.add_parser(
		'expansion_speed',
		help='Guess lineage relations by maximizing the expansion velocity of the bud with respect to the candidate parent.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser_lineage_expspeed.add_argument(
		'--num_frames',
		help='How many frames to consider to compute expansion velocity. At least 2 frames should be considered for good results',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.num_frames,
		type=int
	)
	parser_lineage_expspeed.add_argument(
		'--ignore_dist_nan',
		help='In some cases the computed expansion distance encounters an error (candidate parent flushed away, invalid contour, etc.), then the computed distance is replaced by nan for the given frame. If this happens for many frames, the computed expansion speed might be nan. Enabling this parameter ignores candidates for which the computed expansion speed is nan, otherwise raises an error.',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.ignore_dist_nan,
		type=bool
	)
	parser_lineage_expspeed.add_argument(
		'--bud_distance_max',
		help='Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface"',
		default=bread.algo.lineage._lineage.LineageGuesserExpansionSpeed.bud_distance_max,
		type=int
	)

	parser_lineage_budneck = subparsers_lineage.add_parser(
		'budneck',
		help='Guess lineage relations by looking at the budneck marker intensity along the contour of the bud.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser_lineage_budneck.add_argument(
		'--budneck-file',
		help='Path to file containing budneck marker. Supported file types : tiff (as exported by Fiji)',
		type=pathlib.Path,
		required=True
	)
	parser_lineage_budneck.add_argument(
		'--kernel_N',
		help='Size of the gaussian smoothing kernel in pixels. larger means smoother intensity curves',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.kernel_N,
		type=int,
	)
	parser_lineage_budneck.add_argument(
		'--kernel_sigma',
		help='Number of standard deviations to consider for the smoothing kernel',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.kernel_sigma,
		type=int,
	)
	parser_lineage_budneck.add_argument(
		'--offset_frames',
		help='Wait this number of frames after bud appears to look at the budneck marker channel',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.offset_frames,
		type=int,
	)
	parser_lineage_budneck.add_argument(
		'--num_frames',
		help='Number of frames to watch the budneck marker channel for. The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy',
		default=bread.algo.lineage._lineage.LineageGuesserBudLum.num_frames,
		type=int,
	)

	# Tracker

	parser_tracker = subparsers_main.add_parser(
		'tracker',
		help='Track cells in a movie.'
	)

	# Do stuff

	args = parser_main.parse_args()
	logger.debug(args)

	if args.action == 'lineage':
		if args.lineage_algo == 'expansion_speed':
			from bread.algo.lineage import LineageGuesserExpansionSpeed
			from bread.data import Segmentation, Lineage

			logger.info('Loading segmentation...')
			segmentation = Segmentation.from_h5(args.segmentation_file)
			logger.info(f'Loaded segmentation {segmentation}')

			logger.info('Loading guesser...')
			guesser = LineageGuesserExpansionSpeed(
				segmentation=segmentation,
				nn_threshold=args.nn_threshold,
				flexible_nn_threshold=args.flexible_nn_threshold,
				num_frames=args.num_frames,
				ignore_dist_nan=args.ignore_dist_nan,
				bud_distance_max=args.bud_distance_max
			)
			logger.info(f'Loaded guesser {guesser}')

			logger.info(f'Running guesser...')
			lineage_guess: Lineage = guesser.guess_lineage()
			
			logger.info(f'Saving lineage...')
			lineage_guess.save_csv(args.output_file)

		elif args.lineage_algo == 'budneck':
			from bread.algo.lineage import LineageGuesserBudLum
			from bread.data import Segmentation, Lineage, Microscopy

			logger.info('Loading segmentation...')
			segmentation = Segmentation.from_h5(args.segmentation_file)
			logger.info(f'Loaded segmentation {segmentation}')

			logger.info('Loading budneck channel movie...')
			budneck_img = Microscopy.from_tiff(args.budneck_file)
			logger.info(f'Loaded budneck channel movie {budneck_img}')

			logger.info('Loading guesser...')
			guesser = LineageGuesserBudLum(
				segmentation=segmentation,
				budneck_img=budneck_img,
				nn_threshold=args.nn_threshold,
				flexible_nn_threshold=args.flexible_nn_threshold,
				kernel_N=args.kernel_N,
				kernel_sigma=args.kernel_sigma,
				offset_frames=args.offset_frames,
				num_frames=args.num_frames
			)
			logger.info(f'Loaded guesser {guesser}')

			logger.info(f'Running guesser...')
			lineage_guess: Lineage = guesser.guess_lineage()
			
			logger.info(f'Saving lineage...')
			lineage_guess.save_csv(args.output_file)

	elif args.action == 'tracker':
		raise NotImplementedError()