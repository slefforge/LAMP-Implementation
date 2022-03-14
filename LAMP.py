import shrinkbench
from shrinkbench.experiment import PruningExperiment
from IPython.display import clear_output
import os
from shrinkbench.plot import df_from_results, plot_df
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def prune():
    os.environ['DATAPATH'] = './data'
    for strategy in ['LAMP']: #['RandomPruning', 'GlobalMagWeight', 'LayerMagWeight']:
        print(strategy)
        for  c in [4]: #[1,2,4,8,16,32,64]:
            exp = PruningExperiment(dataset='CIFAR10', 
                                    model='resnet20',
                                    strategy=strategy,
                                    compression=c,
                                    train_kwargs={'epochs':1})
            exp.run()
            clear_output()
    df = df_from_results('results')
    df.to_pickle("./results.pkl")  

def evaluate():
    df = pd.read_pickle("./results.pkl")  
    plot_df(df, 'compression', 'pre_acc5', markers='strategy', line='--', colors='strategy', suffix=' - pre')
    plot_df(df, 'compression', 'post_acc5', markers='strategy', fig=False, colors='strategy')

    plot_df(df, 'speedup', 'post_acc5', colors='strategy', markers='strategy')
    # plt.yscale('log')
    plt.ylim(0.996,0.9995)
    plt.xticks(2**np.arange(7))
    plt.gca().set_xticklabels(map(str, 2**np.arange(7)))
    plt.show()

    df['compression_err'] = (df['real_compression'] - df['compression'])/df['compression']
    plot_df(df, 'compression', 'compression_err', colors='strategy', markers='strategy')

    plt.show()
if __name__ == "__main__":
    print("initiating pruning")
    prune()
    #evaluate()