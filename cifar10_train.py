import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import pickle
import torch
import torch.nn as nn
from cnn import myCNN2
from func import train
from data_io import read_image_list


### データセットに応じてこの部分を書き換える必要あり ###

# 使用するデータセット
DATA_DIR = './dataset/CIFAR10/' # データフォルダのパス（別データセットを使う場合，ここを書き換える）
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ

# 入力画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，下の三つも併せて書き換える）
WIDTH = 32 # CIFAR10物体画像の場合，横幅は 32 pixels
HEIGHT = 32 # CIFAR10物体画像の場合，縦幅も 32 pixels
CHANNELS = 3 # CIFAR10物体画像はカラー画像なので，チャンネル数は 3

### ここまで ###


# 一時ファイルを保存するフォルダの指定
MODEL_DIR = './cifar10_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for CIFAR10 Image Recognition (Training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--model', '-m', default='cifar10_model.pth', type=str, help='file path of trained model')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    epochs = max(1, args.epochs) # 総エポック数（繰り返し回数）
    batchsize = max(1, args.batchsize) # バッチサイズ
    model_filepath = args.model # 学習結果のモデルの保存先ファイル
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('epochs: {0}'.format(epochs), file=sys.stderr)
    print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 学習データの読み込み
    labels, imgfiles, labeldict, labelnames = read_image_list(IMAGE_LIST, DATA_DIR)

    # 情報を一時ファイルに保存（学習後，性能テスト時に使用する）
    n_classes = len(labeldict) # クラス数
    with open(MODEL_DIR + 'labeldict.pickle', 'wb') as f:
        pickle.dump(labeldict, f) # ラベル名からラベル番号を与える辞書をファイルに保存
    with open(MODEL_DIR + 'labelnames.pickle', 'wb') as f:
        pickle.dump(labelnames, f) # ラベル番号からラベル名を与える配列をファイルに保存

    # ネットワークモデルの作成
    cnn = myCNN2(WIDTH, HEIGHT, n_classes)


    ### ここから下は，必要に応じて書き換えると良い ###

    # 追加条件の指定
    conditions = {}
    conditions['in_channels'] = CHANNELS # 入力画像のチャンネル数が 3（カラー画像）

    # 学習処理
    cnn = train(
        device=dev, # 使用するGPUのID，変えなくて良い
        model_dir=MODEL_DIR, # 一時ファイルを保存するフォルダ，変えなくて良い
        in_data=imgfiles, # モデルの入力データ，今回はMNIST画像
        out_data=labels, # モデルの出力データ，今回はクラスラベル
        model=cnn, # 学習するネットワークモデル
        loss_func=nn.CrossEntropyLoss(), # 損失関数，今回は softmax cross entropy
        batchsize=batchsize, # バッチサイズ
        epochs=epochs, # 総エポック数
        stepsize=10, # 高速化のため，ミニバッチ10個につき1個しか学習に用いないことにする
        additional_conditions=conditions # 上で指定した追加条件
    )

    ### ここまで ###


    # 学習結果のモデルをファイルに保存
    torch.save(cnn.state_dict(), model_filepath)

    print('', file=sys.stderr)
