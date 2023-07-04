import pandas as pd
from matplotlib import pyplot as plt


def loss_visualize(epoch_loss, Value1_loss, Value2_loss):

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Loss") #図のタイトル
    plt.plot(epoch_loss, Value1_loss, label='train', color='g',
             linestyle='-')#線1の名前

    plt.plot(epoch_loss, Value2_loss, label='val', color='r', linestyle='-')#線２の名前


    plt.legend()
    plt.xlabel('Epoch')#X軸のタイトル
    plt.ylabel('Loss')#Y軸のタイトル
    plt.grid()
    plt.savefig('./results.png')
    plt.show()


def read_value(train_df):
    epoch = train_df['Step']
    Value1, Value2 = train_df['Value1'], train_df['Value2']
    return epoch, Value1, Value2


if __name__ == "__main__":

    loss = pd.read_csv('C:/Users/user/Desktop/vgg16model/VGG16/vgg_result/logs/tensorboard/csv (80).csv', encoding='shift-jis')  #この.pyファイルがCSVのファイルと同じパスにおいて

    epoch_loss, Value1_loss, Value2_loss = read_value(loss)

    loss_visualize(epoch_loss, Value1_loss, Value2_loss)