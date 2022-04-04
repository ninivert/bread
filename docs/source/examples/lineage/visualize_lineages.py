import pandas as pd
from bread.algo.lineage import align_lineages
from bread.data import Lineage

def highlight_diff(row):
	highlight_same = 'background-color: lightgreen; color: black;'
	highlight_diff = 'background-color: coral; color: black;'
	highlight_root = ''
	highlight_noguess = 'background-color: yellow; color: black;'
	
	if row['ParentID (truth)'] == Lineage.SpecialParentIDs.PARENT_OF_ROOT.value:
		return [highlight_root, highlight_root]

	if row['ParentID (predicted)'] == Lineage.SpecialParentIDs.NO_GUESS.value:
		return [highlight_noguess, highlight_noguess]
	
	if row['ParentID (truth)'] == row['ParentID (predicted)']:
		return [highlight_same, highlight_same]
	else:
		return [highlight_diff, highlight_diff]

def visualize_lineages(lineage_truth, lineage_pred):
	parent_ids, parent_ids_pred, bud_ids, time_ids = align_lineages(lineage_truth, lineage_pred)
	df = pd.DataFrame({
			'ParentID (truth)': parent_ids,
			'ParentID (predicted)': parent_ids_pred,
			'BudID': bud_ids,
			'FrameID': time_ids
		},
		# dtype='Int64'
	)

	return df.style\
		.apply(highlight_diff, subset=['ParentID (truth)', 'ParentID (predicted)'], axis=1)\
		.hide_index()