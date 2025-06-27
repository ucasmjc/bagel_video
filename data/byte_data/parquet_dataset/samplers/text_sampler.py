"""
Text samplers.
"""

from bs4 import BeautifulSoup
import urllib.parse as ul
import html
import ftfy
import re
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Union


class TextSamplerOutput(NamedTuple):
    """
    Return keys for text embedding,
    and optionally additional information to return to user.
    """

    keys: Union[str, List[str]]


class TextSampler(ABC):
    """
    Text sampler base class.

    Child class must implement __call__ method to return the embedding keys.
    Or raise if the text cannot be sampled (key does not exist.)
    """

    @abstractmethod
    def __call__(self, text: Dict[str, str]) -> TextSamplerOutput:
        raise NotImplementedError


class TextAllSampler(TextSampler):
    """
    All text sampler. Returns all texts.

    e.g.
    text_sampler:
      type: all
    """

    def __init__(
        self,
        all: List[str] = None,
        **kwargs,
    ):
        self.all = all

    def __call__(self, text: Dict[str, str]) -> TextSamplerOutput:
        assert len(text) > 0, "The input text does not exist."

        # Get keys.
        keys = list(text.keys())

        if self.all is not None:
            keys = [key for key in self.all if key in keys]
            assert len(
                keys) > 0, f"No valid text sample found under keys: {text.keys()}."

        return TextSamplerOutput(keys=keys)


class TextFrequencySampler(TextSampler):
    """
    Sample text based on frequency.

    e.g.
    text_sampler:
      type: frequency
      frequency:
        no_title_qwen_caption_en_v2_text: 0.9
        no_title_qwen_caption_en_text: 0.9
        origin_caption: 0.1

        # support regular expression
        -----
        .+qwen_caption_en.+: 0.95
        origin_caption: 0.05
        -----
        .+caption_qwen_recaption_cn_long_2_82_text: 0.9
        .+caption_qwen_recaption_cn_2_95_text: 0.9
        origin_caption: 0.1
        -----
    """

    def __init__(
        self,
        frequency: Dict[str, float] = {},
    ):
        self.frequency = frequency
        # Get regular expression.
        self.patterns = (
            {k: re.compile(k) for k in frequency.keys()
             } if frequency is not None else None
        )

    def __call__(self, text: Dict[str, str]) -> TextSamplerOutput:

        assert len(text) > 0, "The input text does not exist."

        # Get keys.
        keys = list(text.keys())

        # Get weights.
        if self.frequency is None or len(self.frequency) == 0:
            weights = np.array([1.0] * len(keys))
        else:
            matchs = {k: (False, "") for k in text.keys()}
            counter = {k: 0 for k in self.frequency.keys()}
            for k in keys:
                for pstr, pat in self.patterns.items():
                    if pat.match(k) is not None:
                        matchs[k] = (True, pstr)
                        counter[pstr] += 1
                        break
            weights = np.array(
                [
                    self.frequency[matchs[k][1]] /
                    counter[matchs[k][1]] if matchs[k][0] else 0.0
                    for k in keys
                ]
            )
        weights_sum = weights.sum()
        assert weights_sum > 0, f"No valid text sample found under keys: {keys}."
        weights /= weights_sum

        # Sample key.
        keys = str(np.random.choice(keys, p=weights))
        return TextSamplerOutput(keys=keys)


class TextPrioritySampler(TextSampler):
    """
    Sample text based on priority.

    e.g.
    text_sampler:
      type: priority
      priority:
        - no_title_qwen_caption_en_v2_text
        - no_title_qwen_caption_en_text
        - origin_caption
    """

    def __init__(
        self,
        priority: List[str] = [],
    ):
        self.priority = priority

    def __call__(self, text: Dict[str, str]) -> TextSamplerOutput:

        assert len(text) > 0, "The input text does not exist."

        # Get keys.
        keys = list(text.keys())

        # Get priorities.
        priorities = [key for key in self.priority if key in keys]

        # Select key.
        if priorities:
            keys = priorities[0]
        else:
            keys = str(np.random.choice(keys))

        return TextSamplerOutput(keys=keys)


"""
Text cleaner. Copied from DeepFloyd IF.
(https://github.com/deep-floyd/IF/blob/develop/deepfloyd_if/modules/t5.py#L125)
"""


class TextCleaner:
    """
    Clear up a caption with strange/improper contents
    """

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + r"\)"
        + r"\("
        + r"\]"
        + r"\["
        + r"\}"
        + r"\{"
        + r"\|"
        + "\\"
        + r"\/"
        + r"\*"
        + r"]{1,}"
    )

    def __call__(self, text):
        # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        caption = re.sub("<br>", " ", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa: E501
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa: E501
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa: E501
            # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(self.bad_punct_regex, r" ", caption)
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        # j2d1a2a...
        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()
