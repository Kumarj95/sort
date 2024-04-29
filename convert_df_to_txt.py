import pandas as pd
import json
import sys
import numpy as np
import pickle
if __name__=="__main__":
    args = json.loads(sys.argv[-1]) # args in a dictionary here where it was a argparse.NameSpace in the main code
    df=pd.read_pickle(args['df_to_convert'])
    pickle.HIGHEST_PROTOCOL = 4
    print(args['hdf_path'])
    df.to_hdf(args['hdf_path'], 'df')
    # np.set_printoptions(threshold=sys.maxsize)
    # with open(args['temp_cols_path'], 'w') as f:
    #     for column in df.columns:
    #         f.write(column+"\n")
    # print(df)
    # df.to_csv(args['temp_text_path'], header=True,index=True,  sep=",")
    