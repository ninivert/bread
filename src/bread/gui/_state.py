from qtpy.QtCore import QObject, Signal, Slot
from bread.data import Lineage, Microscopy, Segmentation
from typing import Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from ._utils import clamp

__all__ = ['AppState']

class AppState(QObject):
	update_segmentation_data = Signal(Segmentation)
	update_microscopy_data = Signal(Microscopy)
	update_budneck_data = Signal(Microscopy)
	update_nucleus_data = Signal(Microscopy)
	update_add_lineage_data = Signal(Lineage, Path)
	update_segmentation_opacity = Signal(float)
	update_microscopy_opacity = Signal(float)
	update_budneck_opacity = Signal(float)
	update_nucleus_opacity = Signal(float)
	update_show_ids = Signal(bool)
	update_show_lineage_arrows = Signal(bool)
	update_frame_index = Signal(int)
	update_frames_max = Signal(int)
	update_centered_cellid = Signal(int, int)

	@dataclass
	class AppData:
		parent: 'AppState' = field(repr=False)
		segmentation: Optional[Segmentation] = None
		microscopy: Optional[Microscopy] = None
		budneck: Optional[Microscopy] = None
		nucleus: Optional[Microscopy] = None
		# lineages: List[Lineage] = field(default_factory=list)

		@property
		def frames_max(self) -> int:
			try:
				m = min(map(lambda x: len(x), filter(lambda x: x is not None, [self.segmentation, self.microscopy, self.budneck, self.nucleus])))
			except ValueError:
				# min() arg is an empty sequence
				m = 0
			return m

		@property
		def frames_homogeneous(self) -> bool:
			return len(self.segmentation or []) == len(self.microscopy or []) == len(self.nucleus or [])

	@dataclass
	class AppValues:
		parent: 'AppState' = field(repr=False)
		show_ids: bool = False
		show_lineage_arrows: bool = False
		frame_index: int = 0
		opacity_segmentation: float = 1
		opacity_microscopy: float = 1
		opacity_budneck: float = 1
		opacity_nucleus: float = 1

	def __init__(self, parent: Optional[QObject] = None) -> None:
		super().__init__(parent)

		self.data = AppState.AppData(self)
		self.values = AppState.AppValues(self)

	def __repr__(self) -> str:
		return f'AppState({self.data}, {self.values})'

	@Slot(bool)
	def set_show_ids(self, v: bool) -> None:
		self.values.show_ids = v
		self.update_show_ids.emit(self.values.show_ids)

	@Slot(bool)
	def set_show_lineage_arrow(self, v: bool) -> None:
		self.values.show_lineage_arrows = v
		self.update_show_lineage_arrows.emit(self.values.show_lineage_arrows)

	@Slot(int)
	def set_frame_index(self, index: int) -> None:
		self.values.frame_index = clamp(index, 0, self.data.frames_max-1)
		self.update_frame_index.emit(self.values.frame_index)

	@Slot(float)
	def set_opacity_segmentation(self, opacity: float) -> None:
		self.values.opacity_segmentation = clamp(opacity, 0, 1)
		self.update_segmentation_opacity.emit(self.values.opacity_segmentation)

	@Slot(float)
	def set_opacity_microscopy(self, opacity: float) -> None:
		self.values.opacity_microscopy = clamp(opacity, 0, 1)
		self.update_microscopy_opacity.emit(self.values.opacity_microscopy)

	@Slot(float)
	def set_opacity_budneck(self, opacity: float) -> None:
		self.values.opacity_budneck = clamp(opacity, 0, 1)
		self.update_budneck_opacity.emit(self.values.opacity_budneck)

	@Slot(float)
	def set_opacity_nucleus(self, opacity: float) -> None:
		self.values.opacity_nucleus = clamp(opacity, 0, 1)
		self.update_nucleus_opacity.emit(self.values.opacity_nucleus)

	@Slot(Segmentation)
	def set_segmentation_data(self, segmentation: Optional[Segmentation]) -> None:
		self.data.segmentation = segmentation
		self.update_segmentation_data.emit(self.data.segmentation)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_microscopy_data(self, microscopy: Optional[Microscopy]) -> None:
		self.data.microscopy = microscopy
		self.update_microscopy_data.emit(self.data.microscopy)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_budneck_data(self, budneck: Optional[Microscopy]) -> None:
		self.data.budneck = budneck
		self.update_budneck_data.emit(self.data.budneck)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Microscopy)
	def set_nucleus_data(self, nucleus: Optional[Microscopy]) -> None:
		self.data.nucleus = nucleus
		self.update_nucleus_data.emit(self.data.nucleus)
		self.update_frames_max.emit(self.data.frames_max)

	@Slot(Lineage, Path)
	def add_lineage_data(self, lineage: Lineage, filepath: Optional[Path] = None) -> None:
		# lineage is not stored, because ownership is in the table
		# maybe table updates should directly update the lineage here ?
		self.update_add_lineage_data.emit(lineage, filepath)

	@Slot(int, int)
	def set_centered_cellid(self, timeid: int, cellid: int) -> None:
		if self.data.segmentation is None or cellid not in self.data.segmentation.cell_ids(timeid):
			return

		self.update_centered_cellid.emit(timeid, cellid)