import os
import shutil
from absl import logging
import tqdm
from nlp.utils import tqdm_wget_pbar
from wget import download
import tarfile
import bs4
import re
from typing import Union, Dict, List, Tuple, Any, Iterator, Callable

URL_IPIPAN = "http://clip.ipipan.waw.pl/NationalCorpusOfPolish\
?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz"


class NKJP:
    """ NKJP class. """
    def __init__(self, dir: str, url: str = None):
        self.headers = None
        self.filter = None
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

        return self

    def parse_headers(self):
        """Creates dictionary of headers."""

        if self.headers is None:
            dirs = os.listdir(self.dir)
            dirs = [x for x in dirs if os.path.isdir(self.dir + "/" + x)]

            logging.info("Parsing headers...")
            headers = []
            for x in tqdm.tqdm(dirs):
                path = self.dir + "/" + x
                header = _parse_header(path)
                header["dir"] = x
                headers.append(header)

            self.headers = headers
        else:
            logging.info("Corpus headers already parsed. Reusing...")

        return self

    SentenceIterator = Iterator[List[str]]
    TextIterator = Iterator[SentenceIterator]

    # TODO: decide whether we need filter here
    def tokenized_sentences(
                            self,
                            filter: Callable[[Dict[str, Any]], bool]=None
                            ) -> TextIterator:
        if self.headers is None:
            logging.error("Corpus not fully initialized. Parse headers first!")
        else:
            for text_dict in self.headers:
                folder = text_dict["dir"]
                if filter is not None:
                    if filter(text_dict):
                        yield text_dict, parse_text(self.dir + "/" + folder)
                else:
                    yield text_dict, parse_text(self.dir + "/" + folder)

    def sentences(self,
                  filter: Callable[[Dict[str, Any]],
                  bool]=None) -> Iterator[str]:

        if self.headers is None:
            logging.error("Corpus not fully initialized. Parse headers first!")
        else:
            for text_dict in self.headers:
                folder = text_dict["dir"]
                if filter is not None:
                    if filter(text_dict):
                        yield text_dict, parse_sentences(self.dir + "/" + folder)
                else:
                    yield text_dict, parse_sentences(self.dir + "/" + folder)

    def texts(self):
        pass


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
        res = [
            note for note, type in zip(notes, types)
            if type == "text_origin"]
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


def _parse_text(dir: str) -> List[Tuple[str, str]]:

    if os.path.isdir(dir):
        with open(dir + "/text.xml") as f:
            text = bs4.BeautifulSoup(f, features="html.parser")
    else:
        "TODO: test behavior!"
        raise Exception("Directory does not exist!")

    res = []
    for tag in ["ab", "u", "p", "head"]:
        sents = text.teicorpus.tei.body.find_all(tag)
        for s in sents:
            res.append((s.get("xml:id"), s.text))

    return res


Segment = Tuple[str, str, int, int]
Sentence = List[Segment]


def _parse_segments(dir: str) -> List[Sentence]:

    if os.path.isdir(dir):
        with open(dir + "/ann_segmentation.xml") as f:
            segments = bs4.BeautifulSoup(f, features="html.parser")
    else:
        raise Exception("Supplied path is not a valid directory!")

    sents = segments.teicorpus.tei.body.find_all("s")
    res = []
    for sent in sents:
        segs = sent.find_all("seg")
        seg_list = []
        for s in segs:
            seg_list.append((s.get("corresp"), s.get("xml:id")))
        res.append(seg_list)

    def _get_token(x: Tuple[str, str]) -> Tuple[str, str, int, int]:
        token_id = x[1]
        xx = re.findall("\(.*\)", x[0])[0]
        sentence_id, start, stop = xx.strip("()").split(",")
        return (sentence_id, token_id, int(start), int(stop))

    res = list(map(lambda x: list(map(_get_token, x)), res))
    return res


def _transform_list_of_tuples_to_dict(x: List[Tuple[Any, ...]]) \
                        -> Dict[str, List[Tuple[str, int, int]]]:
    d = {}
    for key, values in x:
        tt = d.setdefault(key, "")
        for value in values:
            tt += value
        d[key] = tt

    return d


# TODO: change to parse tokenised sentence
def parse_text(dir: str) -> Iterator[List[str]]:

    def _get_tokens(sent: Sentence, texts: Dict[str, str]) -> List[str]:

        res = []
        for txt, _, start, nchars in sent:
            stop = start + nchars
            res.append(texts[txt][start:stop])

        return res

    texts = _parse_text(dir)
    texts = _transform_list_of_tuples_to_dict(texts)

    segs = _parse_segments(dir)

    for sentence in segs:
        yield _get_tokens(sentence, texts)


def parse_sentences(dir: str) -> Iterator[List[str]]:

    def _get_sentence(sent: Sentence, texts: Dict[str, str]) -> Tuple[int, int]:

        res = []
        token_starts = []
        token_stops = []
        for txt, _, start, nchars in sent:
            token_starts.append(start)
            token_stops.append(start + nchars)

        sent_start = min(token_starts)
        sent_stop = max(token_stops)

        return texts[txt][sent_start:sent_stop]

    texts = _parse_text(dir)
    texts = _transform_list_of_tuples_to_dict(texts)

    segs = _parse_segments(dir)

    for sentence in segs:
        yield _get_sentence(sentence, texts)
