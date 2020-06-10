import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from autoencoders import myAutoEncoder
from data_io import read_image_list, load_single_image, load_images
from utils import save_progress


# データセットの指定（別データセットを使う場合，ここを書き換える）
DATA_DIR = './dataset/MNIST/' # データフォルダのパス
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# モデルを保存するフォルダの指定
MODEL_DIR = './compress_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Compression (Training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    parser.add_argument('--feature_dimension', '-f', default=32, type=int, help='dimension of feature space')
    parser.add_argument('--model', '-m', default='compress_model.pth', type=str, help='file path of trained model')
    parser.add_argument('--autosave', '-s', help='automatically save the model in each epoch', action='store_true')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    epochs = max(1, args.epochs) # 総エポック数（繰り返し回数）
    batchsize = max(1, args.batchsize) # バッチサイズ
    dimension = max(1, args.feature_dimension) # 圧縮後のベクトルの次元数
    model_filepath = args.model # 学習結果のモデルの保存先ファイル
    autosave = 'on' if args.autosave else 'off' # 各エポックでモデルを自動保存するか否か
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('epochs: {0}'.format(epochs), file=sys.stderr)
    print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    print('feature dimension: {0}'.format(dimension), file=sys.stderr)
    print('model file: {0}'.format(model_filepath), file=sys.stderr)
    print('autosave mode: {0}'.format(autosave), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定（別データセットを使う場合，ここを書き換える）
    width = 28 # MNIST文字画像の場合，横幅は 28 pixels
    height = 28 # MNIST文字画像の場合，縦幅も 28 pixels
    channels = 1 # MNIST文字画像はグレースケール画像なので，チャンネル数は 1
    color_mode = 0 if channels == 1 else 1

    # 学習データの読み込み
    labels, imgfiles, labeldict, labelnames = read_image_list(IMAGE_LIST, DATA_DIR)
    n_samples = len(imgfiles) # 学習データの総数

    # 評価用データの読み込み
    labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict)
    n_samples_ev = len(imgfiles_ev) # 評価用データの総数

    # 次元圧縮に用いるオートエンコーダの作成
    model = myAutoEncoder(width, height, channels, dimension)
    model = model.to(dev)

    # 損失関数の定義
    loss_func = nn.L1Loss() # mean absolute error

    # オプティマイザーの用意
    optimizer = optim.Adam(model.parameters())

    # 学習処理ループ
    perm = np.random.permutation(n_samples_ev)
    g = load_images(imgfiles_ev, ids=perm[: batchsize], mode=color_mode) # 評価用データを読み込む
    if autosave == 'on':
        save_progress(MODEL_DIR + 'original.png', g, mode=color_mode) # 比較用に評価用データを保存
    for e in range(epochs):

        # 現在のエポック番号を表示
        print('Epoch {0}'.format(e + 1), file=sys.stderr)

        # 損失関数の値が小さくなるように識別器のパラメータを更新
        model.train()
        n_input = 0
        sum_loss = 0
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batchsize * 10): # 高速化のため，ミニバッチ10個につき1個しか学習に用いないことにする
            model.zero_grad()
            x = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], mode=color_mode), device=dev) # 学習データを読み込む
            loss = loss_func(model(x), x)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            n_input += len(x)
            del loss
            del x

        # 損失関数の現在値を表示
        print('  train loss = {0:.6f}'.format(sum_loss / n_input), file=sys.stderr)

        # 評価用データに対する復元結果を保存
        model.eval()
        x = torch.tensor(g, device=dev)
        y = model(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        if autosave == 'on':
            save_progress(MODEL_DIR + 'reconstructed_ep{0}.png'.format(e + 1), y_cpu, mode=color_mode) # 復元結果を保存
        print('  test loss = {0:.6f}'.format(float(loss_func(y, x))), file=sys.stderr)
        del y_cpu
        del y
        del x

        # 現在のモデルをファイルに自動保存
        if autosave == 'on':
            torch.save(model.to('cpu').state_dict(), MODEL_DIR + 'model_ep{0}.pth'.format(e + 1))
            model = model.to(dev)

        print('', file=sys.stderr)

    # 最終結果のモデルをファイルに保存
    torch.save(model.to('cpu').state_dict(), model_filepath)

    print('', file=sys.stderr)
