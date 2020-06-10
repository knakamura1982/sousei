import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import csv
import cv2
import argparse
import numpy as np
import torch
from autoencoders import myAutoEncoder
from data_io import load_single_image
from utils import show_image


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Compression (Execution)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--compress', '-c', help='execute with compression mode', action='store_true')
    parser.add_argument('--decompress', '-d', help='execute with decompression mode', action='store_true')
    parser.add_argument('--feature_dimension', '-f', default=32, type=int, help='dimension of feature space')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input file path')
    parser.add_argument('--out_filepath', '-o', default='', type=str, help='output file path')
    parser.add_argument('--model', '-m', default='', type=str, help='file path of trained model')
    args = parser.parse_args()

    # コマンドライン引数のチェック
    if args.in_filepath is None or args.in_filepath == '':
        print('error: no input file path is specified.', file=sys.stderr)
        exit()
    if args.out_filepath is None or args.out_filepath == '':
        print('error: no output file path is specified.', file=sys.stderr)
        exit()
    if args.model is None or args.model == '':
        print('error: model file is not specified.', file=sys.stderr)
        exit()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    in_filepath = args.in_filepath # 入力ファイルパス
    out_filepath = args.out_filepath # 出力ファイルパス
    dimension = max(1, args.feature_dimension) # 圧縮後のベクトルの次元数
    model_filepath = args.model # 学習済みモデルのファイルパス
    mode = 'decompression' if args.decompress else 'compression'
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('feature dimension: {0}'.format(dimension), file=sys.stderr)
    print('execution mode: {0}'.format(mode), file=sys.stderr)
    print('input file: {0}'.format(in_filepath), file=sys.stderr)
    print('output file: {0}'.format(out_filepath), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，ここを書き換える）
    width = 28 # MNIST文字画像の場合，横幅は 28 pixels
    height = 28 # MNIST文字画像の場合，縦幅も 28 pixels
    channels = 1 # MNIST文字画像はグレースケール画像なので，チャンネル数は 1
    color_mode = 0 if channels == 1 else 1

    # 圧縮／復元に用いるオートエンコーダをロード
    model = myAutoEncoder(width, height, channels, dimension)
    model.load_state_dict(torch.load(model_filepath))
    model = model.to(dev)
    model.eval()

    # 処理の実行
    if mode == 'compression':

        ### 圧縮モード ###

        # 画像を読み込む
        img = np.asarray([load_single_image(in_filepath, mode=color_mode)]) # 入力画像を読み込む
        show_image(img[0], title='input image', mode=color_mode) # 入力画像を表示

        # 圧縮
        x = torch.tensor(img, device=dev)
        y = model.encode(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        del y
        del x

        # 結果を保存
        with open(out_filepath, 'w') as f:
            for i in range(y_cpu.shape[1]):
                print(y_cpu[0, i], file=f)
        del y_cpu

    else:

        ### 復元モード ###

        # 特徴ベクトルファイルを読み込む
        with open(in_filepath, 'r') as f:
            reader = csv.reader(f)
            vec = []
            for row in reader:
                vec.append(float(row[0]))
            vec = np.asarray([vec], dtype=np.float32)

        # 復元
        y = torch.tensor(vec, device=dev)
        x = model.decode(y)
        x_cpu = x.to('cpu').detach().numpy().copy()
        del x
        del y

        # 結果を保存
        show_image(x_cpu[0], title='reconstructed image', mode=color_mode) # 復元画像を表示
        x_cpu = np.asarray(x_cpu[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
        cv2.imwrite(out_filepath, x_cpu)
        del x_cpu

    print('', file=sys.stderr)
