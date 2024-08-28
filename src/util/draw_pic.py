import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

if __name__ == '__main__':

    output = Path(r"D:\Users\hjf\pythoncode\federated\out\sketch\e1")
    for file in output.iterdir():
        if file.suffix=='.csv':
            df = pd.read_csv(file.absolute())
            plt.plot(df.index[:],df['test_correct'],label='acc_'+file.stem)
    plt.legend(title='accuracy')
    plt.savefig(str(output.absolute())+'/accuracy.png')

    plt.close()
    # plt.show()
    for file in output.iterdir():
        if file.suffix=='.csv':
            df = pd.read_csv(file.absolute())
            plt.plot(df.index,df['train_loss'],label=f'{file.stem}_train')
            plt.plot(df.index,df['test_loss'],label=f'{file.stem}_test')
    plt.legend(title='loss')
    plt.savefig(str(output.absolute())+'/loss.png')
    # plt.show()