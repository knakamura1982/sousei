import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import csv
import cv2
import argparse
import numpy as np
import torch
from autoencoders import myUpSamplingAE
from data_io import load_single_image


# 学習済みモデルが保存されているフォルダのパス
MODEL_DIR = './upsampling_models/'

# 学習済みモデルのファイルパス
MODEL_FILE = MODEL_DIR + 'model_ep10.pth'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Up-sampling (Execution)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input file path')
    parser.add_argument('--out_filepath', '-o', default='', type=str, help='output file path')
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
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('input file: {0}'.format(in_filepath), file=sys.stderr)
    print('output file: {0}'.format(out_filepath), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 32 # Place365シーン画像の場合，横幅は 256 pixels であるが，32 pixels となるようにリサイズして使用する
    height = 32 # Place365シーン画像の場合，縦幅も 256 pixels であるが，32 pixels となるようにリサイズして使用する
    channels = 3 # Place365シーン画像はカラー画像なので，チャンネル数は 3
    in_size = (width, height)
    out_size = (width * 2, height * 2)

    # アップサンプリングに用いるオートエンコーダの作成
    model = myUpSamplingAE(width, height, channels)
    model.load_state_dict(torch.load(MODEL_FILE))
    model = model.to(dev)
    model.eval()

    # 処理の実行
    img = np.asarray([load_single_image(in_filepath, size=in_size, mode=1)]) # 入力画像を 32x32 にリサイズして読み込む
    x = torch.tensor(img, device=dev)
    y = model(x)
    y_cpu = y.to('cpu').detach().numpy().copy()
    del y
    del x

    # 結果を保存
    y_cpu = np.asarray(y_cpu[0].transpose(1, 2, 0) * 255, dtype=np.uint8)
    cv2.imwrite(out_filepath, y_cpu)
    del y_cpu

    print('', file=sys.stderr)
