import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from autoencoders import myUpSamplingAE
from data_io import read_image_list
from func import train


### データセットに応じてこの部分を書き換える必要あり ###

# 使用するデータセット
DATA_DIR = './dataset/VGGFace2/' # データフォルダのパス（別データセットを使う場合，ここを書き換える）
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ

# 入力画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，下の二つも併せて書き換える）
WIDTH = 128 # VGGFace2顔画像の場合，横幅は 128 pixels
HEIGHT = 128 # VGGFace2顔画像の場合，縦幅も 128 pixels
CHANNELS = 3 # VGGFace2顔画像はカラー画像なので，チャンネル数は 3

### ここまで ###


# 一時ファイルを保存するフォルダの指定
MODEL_DIR = './upsampling_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Up-sampling (Training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--model', '-m', default='upsampling_model.pth', type=str, help='file path of trained model')
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

    # ネットワークモデルの作成（本来のVGGFace2顔画像を縦横半分にしたものを入力画像として用いるので，WIDTH と HEIGHT を 2 で割る）
    cnn = myUpSamplingAE(WIDTH // 2, HEIGHT // 2, CHANNELS)


    ### ここから下は，必要に応じて書き換えると良い ###

    # 追加条件の指定
    conditions = {}
    conditions['in_channels'] = CHANNELS # 入力画像のチャンネル数が 3（カラー画像）
    conditions['out_channels'] = CHANNELS # 出力画像のチャンネル数も 3（カラー画像）
    conditions['input_scale'] = 0.5 # 本来のVGGFace2顔画像を縦横半分に（0.5倍）して入力画像とする

    # 学習処理
    cnn = train(
        device=dev, # 使用するGPUのID，変えなくて良い
        model_dir=MODEL_DIR, # 一時ファイルを保存するフォルダ，変えなくて良い
        in_data=imgfiles, # モデルの入力データ，今回はVGGFace2顔画像（ただし縦横を半分にして使用）
        out_data=imgfiles, # モデルの出力データ，今回はVGGFace2顔画像
        model=cnn, # 学習するネットワークモデル
        loss_func=nn.L1Loss(), # 損失関数，今回は mean absolute error
        batchsize=batchsize, # バッチサイズ
        epochs=epochs, # 総エポック数
        stepsize=10, # 高速化のため，ミニバッチ10個につき1個しか学習に用いないことにする
        additional_conditions=conditions # 上で指定した追加条件
    )

    ### ここまで ###


    # 最終結果のモデルをファイルに保存
    torch.save(cnn.to('cpu').state_dict(), model_filepath)

    print('', file=sys.stderr)
