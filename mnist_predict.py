import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import pickle
import numpy as np
import torch
from cnn import myCNN
from utils import show_image
from data_io import read_image_list, load_single_image, load_images


# データセットの指定
DATA_DIR = './dataset/MNIST/' # データフォルダのパス
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# 学習済みモデルが保存されているフォルダのパス
MODEL_DIR = './mnist_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for MNIST Image Recognition (Prediction)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--in_filepath', '-i', default='', type=str, help='input image file path')
    parser.add_argument('--model', '-m', default='', type=str, help='file path of trained model')
    args = parser.parse_args()

    # コマンドライン引数のチェック
    if args.model is None or args.model == '':
        print('error: model file is not specified.', file=sys.stderr)
        exit()

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

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 28 # MNIST文字画像の場合，横幅は 28 pixels
    height = 28 # MNIST文字画像の場合，縦幅も 28 pixels
    channels = 1 # MNIST文字画像はグレースケール画像なので，チャンネル数は 1
    color_mode = 0 if channels == 1 else 1

    # ラベル名とラベル番号を対応付ける辞書をロード
    with open(MODEL_DIR + 'labeldict.pickle', 'rb') as f:
        labeldict = pickle.load(f)
    with open(MODEL_DIR + 'labelnames.pickle', 'rb') as f:
        labelnames = pickle.load(f)
    n_classes = len(labelnames)

    # 学習済みの画像認識器をロード
    model = myCNN(width, height, channels, n_classes)
    model.load_state_dict(torch.load(model_filepath))
    model = model.to(dev)
    model.eval()

    # 入力画像に対し認識処理を実行
    if in_filepath == '':

        ### ファイル名を指定せずに実行した場合・・・全評価用データに対する識別精度を表示 ###

        labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict) # 評価用データの読み込み
        n_samples_ev = len(imgfiles_ev) # 評価用データの総数
        n_failed = 0
        for i in range(0, n_samples_ev, batchsize):
            x = torch.tensor(load_images(imgfiles_ev, ids=np.arange(i, i + batchsize), mode=color_mode), device=dev)
            t = labels_ev[i : i + batchsize]
            y = model.classify(x)
            y_cpu = y.to('cpu').detach().numpy().copy()
            n_failed += np.count_nonzero(np.argmax(y_cpu, axis=1) - t)
            del y_cpu
            del y
            del x
            del t
        acc = (n_samples_ev - n_failed) / n_samples_ev
        print('accuracy = {0:.2f}%'.format(100 * acc), file=sys.stderr)

    else:

        ### 画像ファイル名を指定して実行した場合・・・指定された画像に対する認識結果を表示 ###

        img = np.asarray([load_single_image(in_filepath, mode=color_mode)]) # 入力画像を読み込む
        show_image(img[0], title='input image', mode=color_mode) # 入力画像を表示
        x = torch.tensor(img, device=dev)
        y = model.classify(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        print('recognition result: {0}'.format(labelnames[np.argmax(y_cpu)]), file=sys.stderr)
        del y_cpu
        del y
        del x

    print('', file=sys.stderr)
