import sys
import argparse
import urllib.request


URL_HEADER = 'http://www2c.comm.eng.osaka-u.ac.jp/~k-nakamura/temp/others/hptwQMvgxpq60xP2zmW6BD7t/'


# プログレスバー表示用関数（https://qiita.com/jesus_isao/items/ffa63778e7d3952537db より借用）
def progress_print(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = '=' * bar_num
    if bar_num != max_bar:
        progress_element += '>'
    bar_fill = ' '
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        #Fore.LIGHTCYAN_EX,
        f'[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r',
        end=''
    )


# エントリポイント
if __name__ == '__main__':

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description = 'Dataset Downloader')
    parser.add_argument('--target', '-t', default='', type=str, help='target dataset name')
    args = parser.parse_args()

    # コマンドライン引数のチェック
    if args.target is None or args.target == '':
        print('error: target dataset name is not specified.', file=sys.stderr)
        exit()

    # データセットのダウンロード
    target = args.target
    if target == 'MNIST':
        filename = 'MNIST.tar.gz'
    elif target == 'CIFAR10':
        filename = 'CIFAR10.tar.gz'
    elif target == 'Place365':
        filename = 'Place365.tar.gz'
    elif target == 'VGGFace2':
        filename = 'VGGFace2.tar.gz'
    elif target == 'ETL5C':
        filename = 'ETL5C.tar.gz'
    elif target == 'ETL4C':
        filename = 'ETL4C.tar.gz'
    elif target == 'LEGO':
        filename = 'LEGO.tar.gz'
    elif target == 'UTKFace':
        filename = 'UTKFace.tar.gz'
    elif target == 'JAFFE':
        filename = 'JAFFE.tar.gz'
    elif target == 'Food101':
        filename = 'Food101.tar.gz'
    elif target == 'LFW_Sub':
        filename = 'LFW_Sub.tar.gz'
    else:
        print('error: {0} is an invalid dataset name.'.format(target), file=sys.stderr)
        exit()
    url = URL_HEADER + filename
    urllib.request.urlretrieve(url, '{0}'.format(filename), progress_print)
    print('')
    print('')
