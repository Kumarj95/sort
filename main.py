import numpy as np
import pandas as pd
from sort import *
from tqdm import tqdm
import pickle5 as pickle
import motmetrics as mm
from collections import defaultdict, OrderedDict
from glob import glob
import json
import re
import ast
import pathlib
import argparse
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
def conda_pyrun(env_name, exec_file, args):
    os.system(f"conda run -n {env_name} --live-stream python3 \"{exec_file}\" '{json.dumps(dict(vars(args)))}'")

def convert_pandas_versions(args,df_to_convert):
    args.df_to_convert=df_to_convert
    args.hdf_path= os.path.splitext(df_to_convert)[0] +".hdf"
    # args.temp_text_path="./temp_df.txt"
    # args.temp_cols_path='./temp_cols.txt'
    conda_pyrun("pandasver", "convert_df_to_txt.py", args)
    data=pd.read_hdf(args.hdf_path)
    return data
    # df=pd.read_csv(args.temp_text_path, index_col=0, sep=",")
    # df.set_index(df.iloc[:, 0])
    # for column in df.columns:
    #     if( "True"  in str(df[column]) or "False" in str(df[column])):
    #         pattern1= re.compile(r'(?<=[a-zA-Z])\s+(?=[a-zA-Z])')
    #         pattern2=re.compile(r'[\n\\n]')
    #         print(column)
    #         df[column]=df[column].apply(lambda x:re.sub(pattern2,',',re.sub(pattern1,',',x)))
    #         df[column]=df[column].apply(ast.literal_eval)
    #     else:
    #         try:
    #             df[column]=df[column].apply(ast.literal_eval)
                
    #         except SyntaxError as e:
    #             pattern = re.compile(r'(\d)\s+(\d)')
    #             pattern2 = re.compile(r'(\d+)\.(?![0-9])')       
    #             print(df[column])
                    
    #             df[column]=df[column].apply(lambda x:re.sub(pattern,r'\1,\2',re.sub(pattern2,r'\1.0',x)).replace("\n", ','))
    #             print(df[column])
                
    #             df[column]=df[column].apply(ast.literal_eval)
    #         except ValueError as e:
    #             print(e)
    #             pass
    # for index, value in df["trajectory"].iteritems():
    #     print(f"Row {index}: trajectory - {type(value)}") 
    
def main(args):
    max_age=args.MaxAge
    min_hits=args.MinHits
    iou_threshold=args.IouThreshold
    mot_tracker = Sort(max_age, min_hits, iou_threshold/2) #create instance of the SORT tracker
    res_dfs=[[],[],[]]
    for detection, result_path, EdgeIdx, EdgeAttr in zip(args.Detections, args.ResultPath, args.EdgeIdxs, args.EdgeAttributes):
        try:
            with open(detection, "rb") as fh:
                detection_df = pickle.load(fh)
        except AttributeError as e:
                detection_df=convert_pandas_versions(args, detection)
        if(not os.path.exists(result_path[0])):
            pathlib.Path(result_path[0]).parents[0].mkdir(parents=True, exist_ok=True)
        with open(result_path[0], 'w') as f:
            for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
                frame_df = detection_df[detection_df.fn == frame_num]
                # create dets --> this is the part when information is converted/grouped
                dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
                trackers = mot_tracker.update(dets)
                for d in trackers:
                    print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=f) # using frame_num so that trakcing df and detection df are synced
        res_dfs[0].append(df(result_path[0]))
        if(args.UseMethod1):        
            idxs=np.load(EdgeIdx)
            probabilities=np.load(EdgeAttr)
            try:
                with open(detection, "rb") as fh:
                    detection_df = pickle.load(fh)
            except AttributeError as e:
                    detection_df=convert_pandas_versions(args, detection)

            mot_tracker = Sort(max_age, min_hits, iou_threshold/2, df=detection_df, idxs=idxs, probabilities=probabilities, weight=args.Weight) #create instance of the SORT tracker
            if(not os.path.exists(result_path[1])):
                pathlib.Path(result_path[1]).parents[0].mkdir(parents=True, exist_ok=True)
            with open(result_path[1], 'w') as f:
                for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
                    frame_df = detection_df[detection_df.fn == frame_num]
                    # create dets --> this is the part when information is converted/grouped
                    dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=f) # using frame_num so that trakcing df and detection df are synced
            res_dfs[1].append(df(result_path[1]))

        if(args.UseMethod2):        
            with open(args.ParamsPth, 'rb') as f:
                mean=np.load(f)
                var= np.load(f)
            mot_tracker = Sort(max_age, min_hits, iou_threshold, probability_model=(mean,var), weight=args.Weight) #create instance of the SORT tracker
            if(not os.path.exists(result_path[2])):
                pathlib.Path(result_path[2]).parents[0].mkdir(parents=True, exist_ok=True)
            with open(result_path[2], 'w') as f:
                for frame_num in tqdm(range(int(detection_df.fn.min()), int(detection_df.fn.max()+1))): #looping from df.fn.min to df.fn.max
                    frame_df = detection_df[detection_df.fn == frame_num]
                    # create dets --> this is the part when information is converted/grouped
                    dets = frame_df[["x1", "y1", "x2", "y2", "score"]].to_numpy()
                    trackers = mot_tracker.update(dets)
                    for d in trackers:
                        print('%d,%d,%.4f,%.4f,%.4f,%.4f'%(frame_num,d[4],d[0],d[1],d[2],d[3]), file=f) # using frame_num so that trakcing df and detection df are synced
            res_dfs[2].append(df(result_path[2]))
    gt_dfs=[]
    for gt in args.GTs:
        with open(gt, "rb") as fh:
            df_gt = pickle.load(fh)
        gt_dfs.append(df_gt)
    gt_dfs = prepare_df_for_motmetric(gt_dfs, np.arange(len(gt_dfs)))

    for i, res in enumerate(res_dfs):
        if(len(res)>0):
            with open(args.EvaluatePaths[i], 'w') as f:
                pred_dfs = prepare_df_for_motmetric(res, np.arange(len(gt_dfs)))
                accs, names = compare_dataframes(gt_dfs, pred_dfs)
                mh = mm.metrics.create()
                summary = mh.compute_many(accs, names=names, metrics=mm.metrics.motchallenge_metrics, generate_overall=True)

                strsummary = mm.io.render_summary(
                    summary,
                    formatters=mh.formatters,
                    namemap=mm.io.motchallenge_metric_names
                )
                print(strsummary, file=f)

                


def complete_args(args):
    # print(os.path.join(args.Dataset,"*"))
    args.Detections=[]
    args.GTs=[]
    args.ResultPath=[]
    if(args.UseMethod1):
        args.EdgeIdxs=[]
        args.EdgeAttributes=[]
    args.EvaluatePaths=[]

    for d in glob(os.path.join(args.Dataset,"*")):
        if(os.path.isdir(d) and "results" not in d.lower()):
            detection=os.path.join(d,f"{args.VideoName}.detection.{args.Detector}.pkl")
            gt=os.path.join(d,f"{args.VideoName}.tracking.GT.GT.pkl")
            if(args.UseMethod1):
                edgeattr=os.path.join(d,f"{args.VideoName}.tracking.{args.Detector}.{args.Tracker}.EdgeAttr.npy")
                edgeidx=os.path.join(d,f"{args.VideoName}.tracking.{args.Detector}.{args.Tracker}.EdgeIdx.npy")
                args.EdgeIdxs.append(edgeidx)
                args.EdgeAttributes.append(edgeattr)
            args.Detections.append(detection)
            args.GTs.append(gt)
            path_name=os.path.basename(d)
            res_paths=[]
            result_path_1=os.path.join(args.Dataset,"Results",path_name, "result0.txt")
            res_paths.append(result_path_1)
            if(args.UseMethod1):
                result_path_2=os.path.join(args.Dataset,"Results",path_name, "result1.txt")
                res_paths.append(result_path_2) 
            if(args.UseMethod2):
                result_path_3=os.path.join(args.Dataset,"Results",path_name, "result2.txt")
                res_paths.append(result_path_3) 
            args.ResultPath.append(res_paths)
    if(not os.path.exists(os.path.join(args.Dataset,"Results","Evaluate"))):
        os.makedirs(os.path.join(args.Dataset,"Results","Evaluate"), exist_ok=True)
       
    args.EvaluatePaths.append(os.path.join(args.Dataset,"Results","Evaluate","evaluate0.txt"))
    if(args.UseMethod1):
        args.EvaluatePaths.append(os.path.join(args.Dataset,"Results","Evaluate","evaluate1.txt"))
    if(args.UseMethod2):
        args.EvaluatePaths.append(os.path.join(args.Dataset,"Results","Evaluate","evaluate2.txt"))
    return args
if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument("--Dataset", help="Path to repo in specified format for inference (described in readme)", type=str, required=True)
    parser.add_argument("--Weight", help="What weight to give for each method", type=float, default=0.2)
    parser.add_argument("--Detector", help="The name of the detector used", type=str, default="InternImage")
    parser.add_argument("--VideoName", help="The name of the Video for which values were computed", type=str, default="video")
    parser.add_argument("--Tracker", help="The name for the tracker (MPNN1 default)", type=str, default="MPNN1")
    parser.add_argument("--ParamsPth", help="Path to the probaility parameters", type=str)
    parser.add_argument("--UseMethod1", help="If to use GNN based method", action="store_false", default=True)
    parser.add_argument("--UseMethod2", help="If to use probability based method", action="store_false", default=True)
    parser.add_argument("--MaxAge", help="Max age of tracks (from sort paper)", default=19, type=int)
    parser.add_argument("--MinHits", help="Minimum hits for a track to count", default=1, type=int)
    parser.add_argument("--IouThreshold", help="Iou Threshold for tracker", default=0.5, type=float)
    # parser.add_argument("--Train", help="If train the model on dataset", action="store_true")
    # parser.add_argument("--SymLink", help="If filepaths are symlinks (allows us to not have to use up data over and over)", action="store_true")
    # parser.add_argument("--Run", help="if run the model on the dataset", action='store_true')
    # parser.add_argument("--ConfigFile", help="Path to reid config file", type=str)
    # parser.add_argument("--ModelWeights", help="path to reid model weights", type=str)
    # parser.add_argument("--GroundPlane", help="If To train/ track (future) on ground plane",default=False, action="store_true")

    args=parser.parse_args()

    args=complete_args(args)
    main(args)


