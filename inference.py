## Sample code to run inference on inputs to figure out how the sort source code works!
import numpy as np
import pandas as pd
from sort import *
from tqdm import tqdm
import pickle5 as pickle
import motmetrics as mm
from collections import defaultdict, OrderedDict

np.random.seed(0)
max_age=19
min_hits=1
iou_threshold=0.5
DetectionPkl="/home/kumar/sort/dataset/c033/video.detection.InternImage.hdf"
idx_pth="/home/kumar/sort/dataset/c033/video.tracking.InternImage.MPNN1.EdgeIdx.npy"
attr_pth="/home/kumar/sort/dataset/c033/video.tracking.InternImage.MPNN1.EdgeAttr.npy"
gt_path="/home/kumar/sort/dataset/c033/video.tracking.GT.GT.pkl"
idx=np.load(idx_pth)
attr_pth=np.load(attr_pth)
# detection_df = pd.read_pickle(DetectionPkl)
# with open(DetectionPkl, "rb") as fh:
#   detection_df = pickle.load(fh)
detection_df=pd.read_hdf(DetectionPkl)
# finish tracking hyperparams
def df(result_path):
    data = {}
    tracks_path = result_path
    tracks = np.loadtxt(tracks_path, delimiter=',')
    data["fn"]    = tracks[:, 0]
    data["id"]    = tracks[:, 1]
    data["x1"]    = tracks[:, 2]
    data["y1"]    = tracks[:, 3]
    data["x2"]    = tracks[:, 4]
    data["y2"]    = tracks[:, 5]
    # data["class"] = tracks[:, 6]
    return pd.DataFrame.from_dict(data)
def prepare_df_for_motmetric(dfs, cam_ids):
    '''
    in this df we assume that df has the following keys
    x1, y1, x2, y2, id, fn
    '''
    labels = defaultdict(list)
    for df, cam_id in zip(dfs, cam_ids):
        raw_list = []
        for i, row in df.iterrows():
            raw_list.append({'FrameId':row.fn, 'Id':int(row.id), 'X':int(float(row.x1)), 'Y':int(float(row.y1)), 'Width':int(float(row.x2-row.x1)), 'Height':int(float(row.y2-row.y1)), 'Confidence':1.0})
        labels[cam_id].extend(raw_list)
    return OrderedDict([(cam_id, pd.DataFrame(rows).set_index(['FrameId', 'Id'])) for cam_id, rows in labels.items()])
def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []
    for k, tsacc in ts.items():
        accs.append(mm.utils.compare_to_groundtruth(gts[k], tsacc, 'iou', distth=0.5))
        names.append(k)
    return accs, names
# detection_df=detection_df[detection_df['fn']<5]
mot_tracker = Sort(max_age, min_hits, 0.25, df=detection_df, idxs=idx, probabilities=attr_pth, weight=0.3) #create instance of the SORT tracker
inp_np= np.random.random()
with open("results.txt", 'w') as outfile:
  for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
      frame_df = detection_df[detection_df.fn == frame_num]
      # create dets --> this is the part when information is converted/grouped
      dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
      trackers = mot_tracker.update(dets)
      for d in trackers:
          print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=outfile) # using frame_num so that trakcing df and detection df are synced

# mot_tracker = Sort(max_age, min_hits, iou_threshold) #create instance of the SORT tracker
# inp_np= np.random.random()
# with open("results2.txt", 'w') as outfile:
#   for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
#       frame_df = detection_df[detection_df.fn == frame_num]
#       # create dets --> this is the part when information is converted/grouped
#       dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
#       trackers = mot_tracker.update(dets)
#       for d in trackers:
#           print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=outfile) # using frame_num so that trakcing df and detection df are synced

with open("params1d.npy", 'rb') as f:
    mean=np.load(f)
    var= np.load(f)

# with open("params1d_old.npy", 'rb') as f:
#     mean2=np.load(f)
#     var2= np.load(f)

mot_tracker = Sort(max_age, min_hits, iou_threshold, probability_model=(mean,var), weight=0.2) #create instance of the SORT tracker
# inp_np= np.random.random()
# with open("results3.txt", 'w') as outfile:
#   for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
#       frame_df = detection_df[detection_df.fn == frame_num]
#       # create dets --> this is the part when information is converted/grouped
#       dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
#       trackers = mot_tracker.update(dets)
#       for d in trackers:
#           print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=outfile) # using frame_num so that trakcing df and detection df are synced

df1= [df("results.txt")]
df2=[df("/home/kumar/sort/dataset/Results/c033/result1.txt")]
df3=[df("/home/kumar/sort/dataset/Results/c033/result0.txt")]

with open(gt_path, "rb") as fh:
  df_gt = pickle.load(fh)
# print(df1)
# print(df2)

gt_dfs = prepare_df_for_motmetric([df_gt], [1])
pred_dfs = prepare_df_for_motmetric(df1, [1])

accs, names = compare_dataframes(gt_dfs, pred_dfs)
mh = mm.metrics.create()
summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)


gt_dfs = prepare_df_for_motmetric([df_gt], [1])
pred_dfs = prepare_df_for_motmetric(df2, [1])

accs, names = compare_dataframes(gt_dfs, pred_dfs)
mh = mm.metrics.create()
summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)


gt_dfs = prepare_df_for_motmetric([df_gt], [1])
pred_dfs = prepare_df_for_motmetric(df3, [1])

accs, names = compare_dataframes(gt_dfs, pred_dfs)
mh = mm.metrics.create()
summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)
