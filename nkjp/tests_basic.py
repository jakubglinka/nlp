import unittest
import sys
import os
import utils
import corpus


class NKJPMethods(unittest.TestCase):

    def test_nkjp(self):
        nkjp = corpus.NKJP("/tmp/nkjp_test/")
        self.assertEqual(nkjp._is_downloaded(), False)

    def test_clean(self):
        nkjp = corpus.NKJP("/tmp/nkjp_test/", url="http://www.google.com")
        nkjp._download()
        nkjp._clean()
        self.assertFalse(nkjp._is_downloaded())
