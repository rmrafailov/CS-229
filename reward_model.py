import argparse
import  numpy as np
import os
import pandas as pd


def main():

    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()



    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    
    T = data_df.shape[0]
    q = np.zeros(T)
    
    q[T-1] = data_df["speed"][T-1]
    
    for i in reversed(range(T-1)):
        q[i] = data_df["speed"][i:min(i+120,T-1)].mean()
    
    data_df["qs"] = q
    
    data_df.to_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    

if __name__ == '__main__':
    main()