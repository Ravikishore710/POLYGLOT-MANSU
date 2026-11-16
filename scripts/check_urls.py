import urllib.request
import urllib.error

urls = [
    "https://github.com/rahulratan12/Polyglot-MANSU",
    "https://huggingface.co/spaces/Ravikishore710/polyglot-mansu-arena",
    "https://huggingface.co/datasets/Ravikishore710/polyglot-mansu-curated",
    "https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire",
    "https://huggingface.co/Ravikishore710/polyglot-mansu-tiny-aya-fire/discussions/2",
    "https://huggingface.co/Ravikishore710/polyglot-mansu-fire-weights-private"
]

for url in urls:
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req)
        print(f"[HTTP {resp.status}] PUBLIC: {url}")
    except urllib.error.HTTPError as e:
        print(f"[HTTP {e.code}] INACCESSIBLE / PRIVATE / ERROR: {url}")
    except Exception as e:
        print(f"[ERROR] {e}: {url}")
