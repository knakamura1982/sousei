import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
from cnn import myCNN2
from utils import show_image
from func import predict, predict_once
from data_io import read_image_list, load_single_image


### データセットに応じてこの部分を書き換える必要あり ###

# 使用するデータセット
DATA_DIR = './dataset/CIFAR10/' # データフォルダのパス（別データセットを使う場合，ここを書き換える）
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# 画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，下の 3 つを書き換える）
WIDTH = 32 # CIFAR10物体画像の場合，横幅は 32 pixels
HEIGHT = 32 # CIFAR10物体画像の場合，縦幅も 32 pixels
CHANNELS = 3 # CIFAR10物体画像はカラー画像なので，チャンネル数は 3

### ここまで ###


# 一時ファイルが保存されているフォルダのパス
MODEL_DIR = './cifar10_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for MNIST Image Recognition (Prediction)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input image file path')
    parser.add_argument('--model', '-m', default='cifar10_model.pth', type=str, help='file path of trained model')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    in_filepath = args.in_filepath # 入力画像のファイルパス
    batchsize = max(1, args.batchsize) # バッチサイズ
    model_filepath = args.model # 学習済みモデルのファイルパス
    print('device: {0}'.format(dev_str), file=sys.stderr)
    if in_filepath == '':
        print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    else:
        print('input file: {0}'.format(in_filepath), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 学習時に一時ファイルに保存した情報をロード
    with open(MODEL_DIR + 'labeldict.pickle', 'rb') as f:
        labeldict = pickle.load(f)
    with open(MODEL_DIR + 'labelnames.pickle', 'rb') as f:
        labelnames = pickle.load(f)
    n_classes = len(labelnames)

    # 学習済みの画像認識器をロード
    cnn = myCNN2(WIDTH, HEIGHT, n_classes)
    cnn.load_state_dict(torch.load(model_filepath))

    # 入力画像に対し認識処理を実行
    if in_filepath == '':

        ### ファイル名を指定せずに実行した場合・・・全評価用データに対する識別精度を表示 ###

        # 評価用データの読み込み
        labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict)

        # 追加条件の指定
        conditions = {}
        conditions['in_channels'] = CHANNELS # 入力画像のチャンネル数が 1（グレースケール画像）

        # 識別精度評価
        predict(
            device=dev, # 使用するGPUのID，変えなくて良い
            in_data=imgfiles_ev, # 入力データ，今回はMNIST画像
            out_data=labels_ev, # 出力（正解）データ，今回はクラスラベル
            model=cnn, # 学習済みネットワークモデル
            loss_func=nn.CrossEntropyLoss(), # 評価用損失関数，今回は softmax cross entropy
            batchsize=batchsize, # バッチサイズ
            additional_conditions=conditions # 上で指定した追加条件
        )

    else:

        ### 画像ファイル名を指定して実行した場合・・・指定された画像に対する認識結果を表示 ###

        # 入力画像のカラーモードを設定
        color_mode = 0 if CHANNELS == 1 else 1

        # 入力画像を読み込む
        img = load_single_image(in_filepath, mode=color_mode)

        # 入力画像を表示
        show_image(img, title='input image', mode=color_mode)

        # 入力画像をモデルに入力
        y = predict_once(device=dev, model=cnn, in_data=img)

        # 認識結果を表示
        print('recognition result: {0}'.format(labelnames[np.argmax(y)]), file=sys.stderr)

    print('', file=sys.stderr)
