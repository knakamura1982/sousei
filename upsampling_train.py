import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from autoencoders import myUpSamplingAE
from data_io import read_image_list, load_single_image, load_images
from utils import save_progress


# データセットの指定
DATA_DIR = './dataset/VGGFace2/' # データフォルダのパス
IMAGE_LIST = DATA_DIR + 'train_list.csv' # 学習データ
IMAGE_LIST_EV = DATA_DIR + 'test_list.csv' # 評価用データ

# モデルを保存するフォルダの指定
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

    # 画像の縦幅・横幅・チャンネル数の設定
    width = 128 # VGGFace2顔画像の場合，横幅は 128 pixels
    height = 128 # VGGFace2顔画像の場合，縦幅も 128 pixels
    channels = 3 # VGGFace2顔画像はカラー画像なので，チャンネル数は 3
    color_mode = 0 if channels == 1 else 1
    in_size = (width // 2, height // 2)

    # 学習データの読み込み
    labels, imgfiles, labeldict, labelnames = read_image_list(IMAGE_LIST, DATA_DIR)
    n_samples = len(imgfiles) # 学習データの総数

    # 評価用データの読み込み
    labels_ev, imgfiles_ev, labeldict, dmy = read_image_list(IMAGE_LIST_EV, DATA_DIR, dic=labeldict)
    n_samples_ev = len(imgfiles_ev) # 評価用データの総数

    # アップサンプリング用オートエンコーダの作成
    model = myUpSamplingAE(width, height, channels)
    model = model.to(dev)

    # 損失関数の定義
    loss_func = nn.L1Loss() # mean absolute error

    # オプティマイザーの用意
    optimizer = optim.Adam(model.parameters())

    # 学習処理ループ
    perm = np.random.permutation(n_samples_ev)
    g_input = load_images(imgfiles_ev, ids=perm[: batchsize], size=in_size, mode=color_mode) # 評価用データを半分のサイズにリサイズして読み込む
    g_origin = load_images(imgfiles_ev, ids=perm[: batchsize], mode=color_mode) # 評価用データを読み込む
    save_progress(MODEL_DIR + 'input.png', g_input, n_data_max=25, n_data_per_row=5, mode=color_mode)
    save_progress(MODEL_DIR + 'original.png', g_origin, n_data_max=25, n_data_per_row=5, mode=color_mode)
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
            x = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], size=in_size, mode=color_mode), device=dev) # 学習データを半分のサイズにリサイズして読み込む
            t = torch.tensor(load_images(imgfiles, ids=perm[i : i + batchsize], mode=color_mode), device=dev) # 学習データを読み込む
            loss = loss_func(model(x), t)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss) * len(x)
            n_input += len(x)
            del loss
            del t
            del x

        # 損失関数の現在値を表示
        print('  train loss = {0:.6f}'.format(sum_loss / n_input), file=sys.stderr)

        # 評価用データに対するアップサンプリング結果を保存
        model.eval()
        x = torch.tensor(g_input, device=dev)
        t = torch.tensor(g_origin, device=dev)
        y = model(x)
        y_cpu = y.to('cpu').detach().numpy().copy()
        save_progress(MODEL_DIR + 'upsampling_ep{0}.png'.format(e + 1), y_cpu, n_data_max=25, n_data_per_row=5, mode=color_mode)
        print('  test loss = {0:.6f}'.format(float(loss_func(y, t))), file=sys.stderr)
        del y_cpu
        del y
        del t
        del x

        # 現在のモデルをファイルに自動保存
        torch.save(model.to('cpu').state_dict(), MODEL_DIR + 'model_ep{0}.pth'.format(e + 1))
        model = model.to(dev)

        print('', file=sys.stderr)

    # 最終結果のモデルをファイルに保存
    torch.save(model.to('cpu').state_dict(), model_filepath)

    print('', file=sys.stderr)
