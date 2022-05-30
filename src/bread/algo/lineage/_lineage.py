from abc import ABC, abstractmethod
from typing import Callable, Optional, List, Tuple
from dataclasses import dataclass, field
import warnings
import numpy as np
import scipy.signal
import scipy.spatial.distance
import scipy.ndimage

from bread.data import Lineage, Microscopy, Segmentation, Ellipse, Contour, BreadException, BreadWarning

__all__ = [
	'LineageGuesser', 'LineageGuesserBudLum', 'LineageGuesserExpansionSpeed', 'LineageGuesserMinDistance', 'LineageGuesserMinTheta',
	'LineageException', 'LineageWarning',
	'NotEnoughFramesException', 'NotEnoughFramesWarning'
]

class LineageException(BreadException):
	pass

class LineageWarning(BreadWarning):
	pass

class NotEnoughFramesException(LineageException):
	def __init__(self, bud_id: int, time_id: int, num_requested: int, num_remaining: int):
		super().__init__(f'Not enough frames in the movie (bud #{bud_id} at frame #{time_id}), requested {num_requested}, but only {num_remaining} remaining (need at least 2).')

class NotEnoughFramesWarning(LineageWarning):
	def __init__(self, bud_id: int, time_id: int, num_requested: int, num_remaining: int):
		super().__init__(f'Not enough frames in the movie (bud #{bud_id} at frame #{time_id}), requested {num_requested}, but only {num_remaining} remaining.')

@dataclass(frozen=True)
class LineageGuesser(ABC):
	"""Construct LineageGuesser

	Parameters
	----------
	seg : Segmentation
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbours. by default 8.0.
	flexible_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	"""

	segmentation: Segmentation
	nn_threshold: float = 8
	flexible_nn_threshold: bool = False
	num_frames_refractory: int = 0
	
	_cellids_refractory: dict = field(init=False, repr=False, default_factory=dict)

	class NoGuessWarning(LineageWarning):
		def __init__(self, bud_id: int, time_id: int, error: Exception):
			super().__init__(f'Unable to determine parent for bud #{bud_id} in frame #{time_id}. Got error {repr(error)}')

	class NoCandidateParentException(LineageException):
		def __init__(self, time_id: int):
			super().__init__(f'No candidate parents have been found for in frame #{time_id}.')

	@abstractmethod
	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""
		raise NotImplementedError()

	def guess_lineage(self, progress_callback: Optional[Callable[[int, int], None]] = None):
		"""Guess the full lineage of a given bud.

		Returns
		-------
		lineage: Lineage
			guessed lineage
		progress_callback: Callable[[int, int]] or None
			callback for progress
		"""

		lineage_init: Lineage = self.segmentation.find_buds()
		bud_ids, time_ids = lineage_init.bud_ids, lineage_init.time_ids
		parent_ids = np.empty_like(bud_ids, dtype=int)
		# bud_ids, time_ids = bud_ids[time_ids > 0], time_ids[time_ids > 0]

		for i, (bud_id, time_id) in enumerate(zip(bud_ids, time_ids)):
			if progress_callback is not None:
				progress_callback(i, len(time_ids))
			if time_id == 0:
				# cells in first frame have no parent
				parent_ids[i] = Lineage.SpecialParentIDs.PARENT_OF_ROOT.value
				continue

			try:
				parent_ids[i] = self.guess_parent(int(bud_id), int(time_id))  # BUGFIX : cast to an int because indexing with numpy.uint64 raises an error
				self._cellids_refractory[parent_ids[i]] = self.num_frames_refractory
			except BreadException as e:
				if isinstance(e, LineageGuesser.NoCandidateParentException):
					# the cell is too far away from any other cells, it does not belong to the colony
					parent_ids[i] = Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value
				else:
					# the guesser could not give a guess
					warnings.warn(LineageGuesser.NoGuessWarning(bud_id, time_id, e))
					parent_ids[i] = Lineage.SpecialParentIDs.NO_GUESS.value

			self._decrement_refractory()

		return Lineage(parent_ids, bud_ids, time_ids)

	def _decrement_refractory(self):
		"""Decrement the time remaining for refractory cells"""

		pop_cell_ids = []

		# decrement cell timers
		for cell_id in self._cellids_refractory.keys():
			self._cellids_refractory[cell_id] -= 1
			if self._cellids_refractory[cell_id] < 0:
				pop_cell_ids.append(cell_id)

		# remove cells which have finished their refractory period
		for cell_id in pop_cell_ids:
			self._cellids_refractory.pop(cell_id)

	def _candidate_parents(self, time_id: int, excluded_ids: Optional[List[int]] = None, nearest_neighbours_of: Optional[int] = None) -> np.ndarray:
		"""Generate a list of candidate parents to consider for budding events.

		Parameters
		----------
		time_id : int
			frame index in the movie.
		excluded_ids : list[int], optional
			Exclude these cell ids from the candidates.
		nearest_neighbours_of : int or None
			Exclude cells which are not nearest neighbours of the cell with id ``nearest_neighbours_of``.
			Cells for which the smallest distance to cell ``nearest_neighbours_of`` is less than ``self._nn_threshold`` are considered nearest neighbours.
			default is `None`.

		Returns
		-------
		candidate_ids : array-like of int
			ids of the candidate parents
		"""

		cell_ids_prev, cell_ids_curr = self.segmentation.cell_ids(time_id-1), self.segmentation.cell_ids(time_id)
		bud_ids = np.setdiff1d(cell_ids_curr, cell_ids_prev)
		candidate_ids = np.setdiff1d(cell_ids_curr, bud_ids)

		# remove the excluded_ids
		if excluded_ids is not None:
			candidate_ids = np.setdiff1d(candidate_ids, excluded_ids)

		# remove refractory cells
		refractory_ids = np.array(list(self._cellids_refractory.keys()))
		candidate_ids = np.setdiff1d(candidate_ids, refractory_ids)

		# nearest neighbours
		if nearest_neighbours_of is not None:
			contour_bud = Contour.from_segmentation(self.segmentation, nearest_neighbours_of, time_id)
			dists = np.zeros_like(candidate_ids, dtype=float)

			for i, parent_id in enumerate(candidate_ids):
				contour_parent = Contour.from_segmentation(self.segmentation, parent_id, time_id)
				dists[i] = self._nearest_points(contour_bud, contour_parent)[-1]

			if any(dists > self.nn_threshold):
				# the cell has nearest neighbours
				candidate_ids = candidate_ids[dists <= self.nn_threshold]
			else:
				# the cell has no nearest neighbours
				if self.flexible_nn_threshold:
					# pick the closest cell
					candidate_ids = candidate_ids[[np.argmin(dists)]]
				else:
					# warn that the cell has no nearest neighbours
					warnings.warn(BreadWarning(f'cell #{nearest_neighbours_of} does not have nearest neighbours with a distance less than {self.nn_threshold}, and flexible_threshold is {self.flexible_nn_threshold}.'))
					candidate_ids = np.array(tuple())

		if len(candidate_ids) == 0:
			# no nearest neighbours
			raise LineageGuesser.NoCandidateParentException(time_id)

		return candidate_ids

	def _nearest_points(self, contour1: Contour, contour2: Contour) -> Tuple[np.ndarray, np.ndarray, float]:
		"""Find the pair of points between two cell contours which minimizes distance

		Parameters
		----------
		contour1 : Contour
			contour of the first cell
		contour2 : Contour
			contour of the second cell

		Returns
		-------
		point1 : numpy.ndarray (shape=(2,), dtype=int)
		point2 : numpy.ndarray (shape=(2,), dtype=int)
		dist : float
			euclidian distance between point1 and point2
		"""

		dist = scipy.spatial.distance.cdist(contour1.data, contour2.data, 'euclidean')
		i, j = np.unravel_index(dist.argmin(), dist.shape)
		nearest_point1 = contour1[i]
		nearest_point2 = contour2[j]
		min_dist = dist.min()
		return nearest_point1, nearest_point2, min_dist

	def _request_frame_range(self, time_id_from: int, time_id_to: int):
		"""Generate a range of valid frames going from time_id_from (inclusive) to time_id_to (exclusive)

		Parameters
		----------
		time_id_from : int
			index of the first frame
		time_id_to : int
			index of the last frame - 1

		Returns
		-------
		frame_range : range
		"""
		
		assert time_id_from < time_id_to, 'time_id_from should be strictly less than time_id_to'
		num_frames = time_id_to - time_id_from
		num_frames_available = min(max(0, len(self.segmentation) - time_id_from), num_frames)
		return range(time_id_from, time_id_from + num_frames_available)


@dataclass(frozen=True)
class _BudneckMixin:
	# dataclass fields without default value cannot appear after data fields with default values
	# this class provides a mixin to add the positional budneck_img argument
	budneck_img: Microscopy


@dataclass(frozen=True)
class _MajorityVoteMixin:
	"""Redefine ``guess_parent`` to use a majority vote. The method uses an abstract method ``_guess_parent_singleframe`` that needs to be implemented by subclasses"""

	num_frames: int = 5
	offset_frames: int = 0

	def __post_init__(self):
		assert self.num_frames > 0, f'num_frames must be strictly greater than zero, got num_frames={self.num_frames}'

	def guess_parent(self, bud_id: int, time_id: int) -> int:		
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		frame_range = self._request_frame_range(time_id + self.offset_frames, time_id + self.offset_frames + self.num_frames)

		if len(frame_range) == 0:
			raise NotEnoughFramesException(bud_id, time_id, self.num_frames, len(frame_range))
		if len(frame_range) < self.num_frames:
			warnings.warn(NotEnoughFramesWarning(bud_id, time_id, self.num_frames, len(frame_range)))

		parent_ids = []
		exceptions = []

		# guess a parent for each frame in [t + offset, t + offset + duration)
		for time_id_ in frame_range:
			try:
				parent_ids.append(self._guess_parent_singleframe(bud_id, time_id_))
			except BreadException as e:
				# raise e
				exceptions.append(e)
				warnings.warn(LineageWarning(f'Unable to determine parent, got exception {repr(e)}. Computation for frame #{time_id_}, bud #{bud_id} skipped.'))

		if len(parent_ids) == 0:
			# test if the exceptions were raised due to no parent candidates
			if all(isinstance(e, LineageGuesser.NoCandidateParentException) for e in exceptions):
				return Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value

			raise LineageException(f'All of the frames studied ({list(frame_range)}) for bud #{bud_id} (at frame #{time_id}) gave an exception.')

		# perform majority vote
		values, counts = np.unique(parent_ids, return_counts=True)
		majority_ids = values[(counts == np.max(counts))]
		majority_id = majority_ids[0]

		# warn if vote is ambiguous, i.e. 2 or more parents have been guessed the maximum number of times
		# IDEA : return closest dist ?
		if len(majority_ids) > 1:
			warnings.warn(BreadWarning(f'Ambiguous vote for frame #{time_id}, bud #{bud_id}. Possible parents : {majority_ids}'))

		return majority_id

	@abstractmethod
	def _guess_parent_singleframe(self, bud_id, time_id):
		pass


@dataclass(frozen=True)
class LineageGuesserBudLum(_MajorityVoteMixin, LineageGuesser, _BudneckMixin):
	"""Guess lineage relations by looking at the budneck marker intensity along the contour of the bud.

	Parameters
	----------
	segmentation : Segmentation
	budneck_img : Microscopy
	nn_threshold : float, optional
		Cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, default 5
		Number of frames to watch the budneck marker channel for.
		The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy.
	offset_frames : int, default 0
		Wait this number of frames after bud appears before guessing parent.
		Useful if the GFP peak is often delayed.
	kernel_N : int, default 30
		Size of the gaussian smoothing kernel in pixels. larger means smoother intensity curves.
	kernel_sigma : int, default 1
		Number of standard deviations to consider for the smoothing kernel.
	"""

	# budneck_img: Microscopy  # see _BudneckMixin
	# num_frames: int = 5  # see _MajorityVoteMixin
	# offset_frames: int = 0
	kernel_N: int = 30
	kernel_sigma: int = 1

	def __post_init__(self):
		assert self.offset_frames >= 0, f'offset_frames must be greater or equal to zero, got offset_frames={self.offset_frames}'
		assert self.segmentation.data.shape == self.budneck_img.data.shape, f'segmentation and budneck imaging must have the same shape, got segmentation data of shape {self.segmentation.data.shape} and budneck marker data of shape {self.budneck_img.data.shape}'
		
	def _guess_parent_singleframe(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent of a bud, using the budneck marker at a certain frame.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		# explicitly exclude the bud from the considered candidates, because we also study frames after the bud was detected
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)
		contours = [Contour.from_segmentation(self.segmentation, candidate_id, time_id) for candidate_id in candidate_ids]
		
		# Compute position of the GFP peak around the contour of the bud
		contour_peak = self._peak_contour_luminosity(bud_id, time_id)
		# Find the min distances between `contour_peak` and all other cell borders
		min_dists = [scipy.spatial.distance.cdist(contour.data, contour_peak[None, :]).min() for contour in contours]

		# Find index of cell closest to the YFP peak -> parent id
		i_min = min_dists.index(min(min_dists))
		parent_id = candidate_ids[i_min]

		return parent_id

	def _contour_luminosity(self, bud_id: int, time_id: int):
		"""Return contour of a cell mask and the corresponding averaged intensity of a marker along the border.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		contour : Contour
			contour of the bud
		luminosities : numpy.ndarray (shape=(N,), dtype=float)
			averaged luminosities in each of the countour points
		"""

		contour = Contour.from_segmentation(self.segmentation, bud_id, time_id)

		t = np.linspace(-3, 3, self.kernel_N, dtype=np.float64)
		bump = np.exp(-0.5*(t/self.kernel_sigma)**2)
		bump /= np.trapz(bump)  # normalize the integral to 1
		kernel = bump[:, np.newaxis] * bump[np.newaxis, :]  # make a 2D kernel

		# Convolve only a small bounding box

		# Find the cell bounding box
		ymin, ymax, xmin, xmax = contour[:, 1].min(), contour[:, 1].max(), contour[:, 0].min(), contour[:, 0].max()
		# Add enough space to fit the convolution kernel on the border
		ymin, ymax, xmin, xmax = ymin-self.kernel_N, ymax+self.kernel_N, xmin-self.kernel_N, xmax+self.kernel_N
		# Clamp indices to prevent out of bound errors (shape[0]-1 instead of shape because ymax (resp. xmax) is inclusive)
		ymin, ymax = np.clip((ymin, ymax), 0, self.budneck_img[time_id].shape[0]-1)
		xmin, xmax = np.clip((xmin, xmax), 0, self.budneck_img[time_id].shape[1]-1)
		# Recover trimmed image
		img_trimmed = self.budneck_img[time_id, ymin:ymax+1, xmin:xmax+1]

		# Perform smoothing

		# Using the scipy fftconvolution
		# mode='same' is there to enforce the same output shape as input arrays (ie avoid border effects)
		convolved = scipy.signal.fftconvolve(img_trimmed, kernel, mode='same')
		luminosities = convolved[contour[:, 1]-ymin, contour[:, 0]-xmin]

		# Using the custom convolution
		# after testing, this is barely faster. see bread.utils.convolve
		# from bread.utils.convolve import convolve2d
		# luminosities = convolve2d(
		# 	np.ascontiguousarray(img_trimmed, dtype=np.float64),
		# 	np.ascontiguousarray(kernel, dtype=np.float64),
		# 	np.ascontiguousarray(contour[:, [1, 0]] - [ymin, xmin], dtype=int)
		# )

		return contour, luminosities

	def _peak_contour_luminosity(self, bud_id: int, time_id: int) -> np.ndarray:
		"""Return pixel index (i, j) of the maximum luminosity around the border.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		contour_peak : numpy.ndarray (shape=(2,))
			pixel index of the maximum luminosity
		"""

		contours, luminosities = self._contour_luminosity(bud_id, time_id)

		i_peak = np.argmax(luminosities)
		contour_peak = contours[i_peak]

		return contour_peak


@dataclass(frozen=True)
class LineageGuesserExpansionSpeed(LineageGuesser):
	"""Guess lineage relations by maximizing the expansion velocity of the bud with respect to the candidate parent.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, optional
		How many frames to consider to compute expansion velocity.
		At least 2 frames should be considered for good results.
		by default 5.
	ignore_dist_nan : bool, optional
		In some cases the computed expansion distance encounters an error (candidate parent flushed away, invalid contour, etc.),
		then the computed distance is replaced by nan for the given frame.
		If this happens for many frames, the computed expansion speed might be nan.
		Enabling this parameter ignores candidates for which the computed expansion speed is nan, otherwise raises an error.
		by default True.
	bud_distance_max : float, optional
		Maximal distance (in pixels) between points on the parent and bud contours to be considered as part of the "budding interface".
		by default 7.
	"""

	num_frames: int = 5
	ignore_dist_nan: bool = True
	bud_distance_max: float = 7

	class BudVanishedException(LineageException):
		def __init__(self, bud_id: int, time_id: int):
			super().__init__(f'Bud #{bud_id} vanished in between frame {time_id-1} to {time_id}.')

	class NanSpeedException(LineageException):
		def __init__(self, bud_id: int, parent_ids: int, time_id: int):
			super().__init__(f'Unable to determine parent for bud #{bud_id} at frame #{time_id}. The following parent candidates caused a problem : {parent_ids}')

	def __post_init__(self):
		assert self.num_frames >= 2, f'not enough consecutive frames considered for analysis, got {self.num_frames}.'

	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""

		candidate_parents = self._candidate_parents(time_id, nearest_neighbours_of=bud_id)
		frame_range = self._request_frame_range(time_id, time_id + self.num_frames)

		if len(frame_range) < 2:
			raise NotEnoughFramesException(bud_id, time_id, self.num_frames, len(frame_range))
		if len(frame_range) < self.num_frames:
			warnings.warn(NotEnoughFramesWarning(bud_id, time_id, self.num_frames, len(frame_range)))

		# check the bud still exists !
		for time_id_ in frame_range:
			if bud_id not in self.segmentation.cell_ids(time_id_):
				raise LineageGuesserExpansionSpeed.BudVanishedException(bud_id, time_id_)

		dists_all = [self._expansion_distances(bud_id, parent_id, time_id) for parent_id in candidate_parents]
		# numpy.gradient can work with nan values
		# dists might contain nan values, we need to use numpy.nanmean to ignore them for the velocity computation
		mean_vels = np.array([np.nanmean(np.gradient(dists)) for dists in dists_all])

		# test for nan values which might have propagated to final velocities (not enough valid distances !)
		if not self.ignore_dist_nan and np.any(np.isnan(mean_vels)) > 0:
			raise LineageGuesserExpansionSpeed.NanSpeedException(bud_id, candidate_parents[np.isnan(mean_vels)], time_id)

		return candidate_parents[np.nanargmax(mean_vels)]

	def _expansion_distances(self, bud_id: int, parent_id: int, time_id: int) -> np.ndarray:
		"""Computes the "expansion distances" of a bud relative to a candidate cell. Wraps ``LineageGuesserExpansionSpeed._expansion_distance``.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		dists: numpy.ndarray
			expansion distances (warning : might contain nan values)
		"""
		
		frame_range = self._request_frame_range(time_id, time_id + self.num_frames)
		dists = np.zeros(len(frame_range), dtype=float)

		for i, time_id_ in enumerate(frame_range):
			try:
				dists[i] = self._expansion_distance(bud_id, parent_id, time_id_)
			except Contour.InvalidContourException as e:
				warnings.warn(LineageWarning(f'Unable to determine distance due to invalid contour, got exception {repr(e)}. Computation for frame #{time_id_}, bud #{bud_id}, candidate #{parent_id} skipped.'))
				dists[i] = np.nan

		return dists

	def _expansion_distance(self, bud_id: int, parent_id: int, time_id: int, contact_point_kwargs: dict = {}) -> float:
		"""Computes the "expansion distance" of a bud relative to a candidate cell.

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		dist: float
		"""

		bud_contour = Contour.from_segmentation(self.segmentation, bud_id, time_id)
		budding_point = self._budding_point(bud_id, parent_id, time_id)
		point_bud_farthest = self._farthest_point_on_contour(budding_point, bud_contour)

		return scipy.spatial.distance.cdist(budding_point[None, :], point_bud_farthest[None, :])[0, 0]

	def _farthest_point_on_contour(self, point: np.ndarray, contour: Contour) -> np.ndarray:
		"""Return the farthest point from ``point`` on a given ``contour``

		Parameters
		----------
		point : np.ndarray, shape=(2,)
		contour : Contour

		Returns
		-------
		point : np.ndarray, shape=(2,)
			the farthest point
		"""

		dist = scipy.spatial.distance.cdist(point[None, :], contour.data, 'euclidean')
		i, j = np.unravel_index(dist.argmax(), dist.shape)
		return contour[j]

	def _budding_point(self, bud_id: int, parent_id: int, time_id: int):
		"""Compute the budding point of a bud with respect to its candidate parent

		Method :
		1. find all the points on the parent's contour which are close enough to the bud (distance less than ``self._bud_distance_max``)
		2. if there are more than 1 points, compute the average of these points ; else select the closest one

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		parent_id : int
			id of the candidate parent in the segmentation
		time_id : int
			frame index in the movie
		Returns
		-------
		budding_point : np.ndarray, shape=(2,)
			the guessed point from which the bud grows
		"""

		contour_parent = Contour.from_segmentation(self.segmentation, parent_id, time_id)
		contour_bud = Contour.from_segmentation(self.segmentation, bud_id, time_id)
		dists = scipy.spatial.distance.cdist(contour_parent.data, contour_bud.data, 'euclidean')
		dists_mask_close = dists < self.bud_distance_max

		if np.any(dists_mask_close):
			# at least one point on the parent-bud interface
			ijs = np.argwhere(dists <= self.bud_distance_max)
			budding_point = np.mean(contour_parent[ijs[:, 0]], axis=0)
		else:
			# no point on the parent-bud interface, resort to finding smallest distance
			i, j = np.unravel_index(dists.argmin(), dists.shape)
			budding_point = contour_parent[i]
			# warnings.warn(f'no distance below {self._bud_distance_max}, found mininimum {dists.min()}')
		
		return budding_point


@dataclass(frozen=True)
class LineageGuesserMinTheta(_MajorityVoteMixin, LineageGuesser):
	"""Guess lineage relations by minimizing the angle between the major axis of the candidates and candidate-to-bud vector.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	num_frames : int, default 5
		Number of frames to make guesses for after the bud has appeared.
		The algorithm makes a guess for each frame, then predicts a parent by majority-vote policy.
	offset_frames : int, default 0
		Wait this number of frames after bud appears before guessing parent.
	"""

	def _guess_parent_singleframe(self, bud_id, time_id):
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)
		candidate_cms = self.segmentation.cms(time_id, candidate_ids)
		bud_cm = self.segmentation.cms(time_id, [bud_id])[0]
		contours = [Contour.from_segmentation(self.segmentation, candidate_id, time_id) for candidate_id in candidate_ids]
		ellipses = [Ellipse.from_contour(contour) for contour in contours]
		maj_axes = [np.array([np.sin(e.angle), np.cos(e.angle)]) for e in ellipses]  # WARNING : centers of mass are (y, x)
		vecs = [bud_cm - candidate_cm for candidate_cm in candidate_cms]
		nvecs = [vec / np.linalg.norm(vec) for vec in vecs]
		abscosthetas = [abs(np.dot(nvec, maj_ax)) for nvec, maj_ax in zip(nvecs, maj_axes)]
		return candidate_ids[abscosthetas.index(max(abscosthetas))]


@dataclass(frozen=True)
class LineageGuesserMinDistance(LineageGuesser):
	"""Guess lineage relations by finding the cell closest to the bud, when it appears on the segmentation.
	
	Parameters
	----------
	segmentation : Segmentation
	nn_threshold : float, optional
		cell masks separated by less than this threshold are considered neighbors, by default 8.0.
	flexible_nn_threshold : bool, optional
		If no nearest neighbours are found within the given threshold, try to find the closest one, by default False.
	num_frames_refractory : int, optional
		After a parent cell has budded, exclude it from the parent pool in the next frames.
		It is recommended to set it to a low estimate, as high values will cause mistakes to propagate in time.
		A value of 0 corresponds to no refractory period.
		by default 0.
	"""

	def guess_parent(self, bud_id: int, time_id: int) -> int:
		"""Guess the parent associated to a bud

		Parameters
		----------
		bud_id : int
			id of the bud in the segmentation
		time_id : int
			frame index in the movie

		Returns
		-------
		parent_id : int
			guessed parent id
		"""
		
		candidate_ids = self._candidate_parents(time_id, excluded_ids=[bud_id], nearest_neighbours_of=bud_id)

		if len(candidate_ids) == 0:
			return Lineage.SpecialParentIDs.PARENT_OF_EXTERNAL.value

		contour_bud = Contour.from_segmentation(self.segmentation, bud_id, time_id)
		dists = np.zeros_like(candidate_ids, dtype=float)

		for i, parent_id in enumerate(candidate_ids):
			contour_parent = Contour.from_segmentation(self.segmentation, parent_id, time_id)
			dists[i] = self._nearest_points(contour_bud, contour_parent)[-1]

		return candidate_ids[np.argmin(dists)]