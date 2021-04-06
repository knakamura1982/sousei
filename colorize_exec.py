import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import csv
import cv2
import argparse
import numpy as np
import torch
from autoencoders import myColorizationAE
from data_io import load_single_image
from utils import show_image
from func import predict_once


### データセットに応じてこの部分を書き換える必要あり ###

# 入力画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，下の二つを書き換える）
WIDTH = 128 # VGGFace2顔画像の場合，横幅は 128 pixels
HEIGHT = 128 # VGGFace2顔画像の場合，縦幅も 128 pixels

### ここまで ###


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Colorization (Execution)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input file path')
    parser.add_argument('--out_filepath', '-o', default='', type=str, help='output file path')
    parser.add_argument('--model', '-m', default='colorize_model.pth', type=str, help='file path of trained model')
    args = parser.parse_args()

    # コマンドライン引数のチェック
    if args.in_filepath is None or args.in_filepath == '':
        print('error: no input file path is specified.', file=sys.stderr)
        exit()
    if args.out_filepath is None or args.out_filepath == '':
        print('error: no output file path is specified.', file=sys.stderr)
        exit()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    in_filepath = args.in_filepath # 入力ファイルパス
    out_filepath = args.out_filepath # 出力ファイルパス
    model_filepath = args.model # 学習済みモデルのファイルパス
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('input file: {0}'.format(in_filepath), file=sys.stderr)
    print('output file: {0}'.format(out_filepath), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 学習済みのオートエンコーダをロード
    cnn = myColorizationAE(WIDTH, HEIGHT)
    cnn.load_state_dict(torch.load(model_filepath))

    # mode=0: 入力ファイルをグレースケール画像として読み込む
    img = load_single_image(in_filepath, mode=0)

    # 入力画像を表示
    show_image(img, title='input image', mode=0)

    # 入力画像をモデルに入力してカラー化
    y = predict_once(device=dev, model=cnn, in_data=img)

    # カラー化結果を表示
    show_image(y, title='output image', mode=1)

    # カラー化結果をファイルに保存
    y = np.asarray(y.transpose(1, 2, 0) * 255, dtype=np.uint8)
    cv2.imwrite(out_filepath, y)

    print('', file=sys.stderr)
