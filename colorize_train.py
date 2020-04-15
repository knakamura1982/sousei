import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from autoencoders import myColorizationAE
from data_io import read_image_list, load_single_image, load_images
from utils import save_progress


# データセットの指定
DATA_DIR = './dataset/VGGFace2/' # データフォルダのパス
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# モデルを保存するフォルダの指定
MODEL_DIR = './colorize_models/'


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'CNN Model for Image Colorization (Training)')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epochs', '-e', default=10, type=int, help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
    args = parser.parse_args()

    # デバイスの設定
    dev_str = 'cuda:{0}'.format(args.gpu) if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    dev = torch.device(dev_str)

    # オプション情報の設定・表示
    epochs = max(1, args.epochs) # 総エポック数（繰り返し回数）
    batchsize = max(1, args.batchsize) # バッチサイズ
    print('device: {0}'.format(dev_str), file=sys.stderr)
    print('epochs: {0}'.format(epochs), file=sys.stderr)
    print('batchsize: {0}'.format(batchsize), file=sys.stderr)
    print('', file=sys.stderr)

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 128 # VGGFace2顔画像の場合，横幅は 128 pixels
    height = 128 # VGGFace2顔画像の場合，縦幅も 128 pixels

    # 学習データの読み込み
    labels, imgfiles, labeldict, labelnames = read_image_list(IMAGE_LIST, DATA_DIR)
    n_samples = len(imgfiles) # 学習データの総数

    # 評価用データの読み込み
    labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict)
    n_samples_ev = len(imgfiles_ev) # 評価用データの総数

    # カラー化オートエンコーダの作成
    model = myColorizationAE(width, height)
    model = model.to(dev)

    # 損失関数の定義
    loss_func = nn.MSELoss() # mean squared error

    # オプティマイザーの用意
    optimizer = optim.Adam(model.parameters())

    # 学習処理ループ
    perm = np.random.permutation(n_samples_ev)
    g_gray = load_images(imgfiles_ev, ids=perm[: batchsize], mode=0) # 評価用データの一部をグレースケール画像として読み込む
    g_color = load_images(imgfiles_ev, ids=perm[: batchsize], mode=1) # 評価用データの一部をカラー画像として読み込む
    save_progress(MODEL_DIR + 'input.png', g_gray, n_data_max=25, n_data_per_row=5, mode=0)
    save_progress(MODEL_DIR + 'original.png', g_color, n_data_max=25, n_data_per_row=5, mode=1)
    for e in range(epochs):

        # 現在のエポック番号を表示
        print('Epoch {0}'.format(e + 1), file=sys.stderr)

        # 損失関数の値が小さくなるように識別器のパラメータを更新
        model.train()
        sum_loss = 0
        perm = np.random.permutation(n_samples)
        for i in range(0, n_samples, batchsize):
            model.zero_grad()
            x = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], mode=0), device=dev) # mode=0: 学習データをグレースケール画像として読み込む
            t = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], mode=1), device=dev) # mode=1: 学習データをカラー画像として読み込む
            loss = loss_func(model(x), t)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            del loss
            del t
            del x

        # 損失関数の現在値を表示
        print('  train loss = {0:.6f}'.format(sum_loss / n_samples), file=sys.stderr)

        # 評価用データに対するカラー化結果を保存
        model.eval()
        x = torch.tensor(g_gray, device=dev)
        t = torch.tensor(g_color, device=dev)
        y = model(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        save_progress(MODEL_DIR + 'colorized_ep{0}.png'.format(e + 1), y_cpu, n_data_max=25, n_data_per_row=5, mode=1)
        print('  test loss = {0:.6f}'.format(float(loss_func(y, t))), file=sys.stderr)
        del y_cpu
        del y
        del t
        del x

        # 現在のモデルをファイルに保存
        torch.save(model.to('cpu').state_dict(), MODEL_DIR + 'model_ep{0}.pth'.format(e + 1))
        model = model.to(dev)

        print('', file=sys.stderr)

    print('', file=sys.stderr)
