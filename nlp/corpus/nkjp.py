import os
import shutil
from absl import logging
import tqdm
from nlp.utils import tqdm_wget_pbar
from wget import download
import tarfile
import bs4
from typing import Union, Dict, List

URL_IPIPAN = "http://clip.ipipan.waw.pl/NationalCorpusOfPolish\
?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz"


class NKJP:
    """ NKJP class. """
    def __init__(self, dir: str, url: str = None):
        if url is not None:
            self.url = url
        else:
            self.url = URL_IPIPAN
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
            file_path = self.dir + "/nkjp.tar.gz"
            logging.info("Extracting {} file...".format(file_path))
            try:
                with tarfile.open(file_path) as tar:
                    tar.extractall(self.dir)
            except:
                logging.error("Unable to extract nkjp...")
                self._clean()

    def download(self):
        self._download()
        self._extract()


def _header_get_words(header: bs4.BeautifulSoup) -> Union[int, None]:
    try:
        return int(header.teiheader.filedesc.extent.num.get("value"))
    except:
        return None


def _header_get_publisher(header: bs4.BeautifulSoup) -> Union[str, None]:
    try:
        return header.teiheader.filedesc.sourcedesc.publisher.text
    except:
        return None


def _header_get_text_origin(header: bs4.BeautifulSoup) -> Union[str, None]:

    notes = header.teiheader.filedesc.sourcedesc.find_all("note")
    if len(notes):
        types = map(lambda x: x.get("type"), notes)
        res = [note for note, type in zip(notes, types) if type == "text_origin"]
        res = res[0]
        res = res.text
    else:
        res = None

    return res


def _header_get_text_class_tags(header: bs4.BeautifulSoup) -> Dict[str, str]:

    refs = header.profiledesc.textclass.find_all("catref")
    return dict([(r.get("scheme"), r.get("target")) for r in refs])


def _parse_header(dir: str) -> Dict[str, Union[str, int]]:

    if os.path.isdir(dir):
        with open(dir + "/header.xml") as f:
            header = bs4.BeautifulSoup(f, features="html.parser")
    else:
        "TODO: test behavior!"
        raise Exception("Directory does not exist!")

    res = {}
    res["publisher"] = _header_get_publisher(header)
    res["words"] = _header_get_words(header)
    res["text_origin"] = _header_get_text_origin(header)
    res.update(_header_get_text_class_tags(header))

    return res


def parse_headers(corpus: NKJP) -> List[Dict[str, Union[str, int]]]:
    """Creates dictionary of headers."""

    dirs = os.listdir(corpus.dir)
    dirs = [x for x in dirs if os.path.isdir(corpus.dir + "/" + x)]

    logging.info("Parsing headers...")
    headers = []
    for x in tqdm.tqdm(dirs):
        path = corpus.dir + "/" + x
        header = _parse_header(path)
        header["dir"] = x
        headers.append(header)

    return headers
