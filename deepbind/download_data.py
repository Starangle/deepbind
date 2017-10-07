from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import os


def download(limit):
    root_url = "http://cnn.csail.mit.edu/motif_discovery/"
    data_prefix = "data/"
    html = urlopen(root_url).read()
    bs_obj = BeautifulSoup(html)
    href_container = bs_obj.find_all("tr")
    if limit > 690:
        limit = 690
    for tr in href_container[3:3 + limit]:
        href = tr.find_all("td")[1].get_text()
        local_url = data_prefix + href
        train_url = root_url + href + "train.data"
        test_url = root_url + href + "test.data"
        if os.path.exists(local_url):
            print("The data " + href[0:-1] + " had existed!")
        else:
            os.mkdir(local_url)
            print("Folder " + href[0:-1] + " create")
            print("Download train.data ...")
            urlretrieve(train_url, local_url + "train.data")
            print("Download test.data ...")
            urlretrieve(test_url, local_url + "test.data")
            print("The data " + href[0:-1] + " download complete!")


def make_list():
    with open("data_list.txt", "w+") as f:
        for name in os.listdir("data"):
            f.write('data\\' + name + "\\")
            f.write("\n")


if __name__ == '__main__':
    download(680)
    make_list()
