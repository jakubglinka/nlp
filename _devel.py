from nlp.corpus import nkjp
import pandas as pd
import absl
absl.logging.set_verbosity(absl.logging.INFO)

corpus = nkjp.NKJP(dir="/tmp/nkjp/")
corpus.download()

headers = nkjp.parse_headers(corpus)

df = pd.DataFrame(headers)
print(df.head(10))

df.describe(include="all")

df["#taxonomy-NKJP-channel"].value_counts()
df["#taxonomy-NKJP-type"].value_counts()

# import bs4
# with open("/tmp/nkjp/610-1-000988/header.xml") as f:
#     header = bs4.BeautifulSoup(f)

# refs = header.profiledesc.textclass.find_all("catref")
# dict([(r.get("scheme"), r.get("target")) for r in refs])

# from nlp.corpus.nkjp import URL_IPIPAN
# URL_IPIPAN
