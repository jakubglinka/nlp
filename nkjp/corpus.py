import tqdm
import os
from utils import tqdm_wget_pbar
from wget import download
import shutil
from absl import logging
import tarfile
import pandas as pd
import bs4


class NKJP:
    """ NKJP class. """
    def __init__(self, dir: str, url: str = None):
        if url is not None:
            self.url = url
        else:
            self.url = "http://clip.ipipan.waw.pl/NationalCorpusOfPolish?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz"
        self.dir = dir

    def _is_downloaded(self) -> bool:
        return os.path.exists(self.dir)

    def _download(self):
        logging.info("Downloading NKJP corpus to {}...".format(self.dir))
        if self._is_downloaded() is False:
            os.mkdir(self.dir)
            with tqdm_wget_pbar() as progress_bar:
                def _pbar(elapsed: int, total: int, done: int):
                    progress_bar.update(elapsed, total, done)
                download(self.url, self.dir + "/nkjp.tar.gz", _pbar)
        else:
            logging.info("Reusing previously cached corpus...")

    def _clean(self):
        logging.info("Removing corpus...")
        shutil.rmtree(self.dir)

    def _is_extracted(self):
        nkjp_folders = [f for f in os.listdir(self.dir) if f != 'nkjp.tar.gz']
        return len(nkjp_folders) > 0

    def _extract(self):
        if self._is_extracted() is False:
            logging.info("Extracting {} file...".format(self.dir + "/nkjp.tar.gz"))
            try:
                with tarfile.open(self.dir + "/nkjp.tar.gz") as tar:
                    tar.extractall(self.dir)
            except:
                logging.error("Unable to extract nkjp...")
                self._clean()

    def download(self):
        self._download()
        self._extract()


def get_nkjp_stats(nkjp: NKJP) -> pd.DataFrame:
    """Gets statistics of the NKJP corpus."""
    return None

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    nkjp = NKJP("/tmp/nkjp/")
    nkjp.download()



# open file:
dir = nkjp.dir + "620-3-010000124"
with open(dir + "/header.xml") as f:
    header = bs4.BeautifulSoup(f)

from typing import Dict, Union, Any
def _parse_header(dir: str) -> Dict[str, Union[str, Any]]:

    return None


def _header_get_words(header: bs4.BeautifulSoup) -> Union[int, None]:
    try:
        return int(header.teiheader.filedesc.extent.num.get("value"))
    except:
        return None

def _header_get_publisher()
def _header_get_publication_date


_header_get_words(header)



header.teiheader.filedesc.extent.num.get("value")
