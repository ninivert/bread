from bread.data import Lineage, Segmentation

__all__ = ['extend_budding_event']

def extend_budding_event(lineage: Lineage, time_index_num, time_index_max=None):
	"""add extra time to a budding event from a lineage, for a total of time_index_num frames"""

	lineage_extended = {'parent_id': [], 'bud_id': [], 'time_index': []}

	for idx, (parent_id, bud_id, time_index) in lineage.iterrows():
		for dt in range(0, time_index_num):
			if time_index_max is not None and time_index + dt > time_index_max: continue
			lineage_extended['parent_id'].append(parent_id)
			lineage_extended['bud_id'].append(bud_id)
			lineage_extended['time_index'].append(time_index+dt)

	return lineage_extended
