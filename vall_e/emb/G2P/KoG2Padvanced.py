__all__ = ["split_syllable_char", "split_syllables",
           "join_jamos", "join_jamos_char",
           "CHAR_INITIALS", "CHAR_MEDIALS", "CHAR_FINALS"]

import itertools

INITIAL = 0x001
MEDIAL = 0x010
FINAL = 0x100
CHAR_LISTS = {
    INITIAL: list(map(chr, [
        0x3131, 0x3132, 0x3134, 0x3137, 0x3138, 0x3139,
        0x3141, 0x3142, 0x3143, 0x3145, 0x3146, 0x3147,
        0x3148, 0x3149, 0x314a, 0x314b, 0x314c, 0x314d,
        0x314e
    ])),
    MEDIAL: list(map(chr, [
        0x314f, 0x3150, 0x3151, 0x3152, 0x3153, 0x3154,
        0x3155, 0x3156, 0x3157, 0x3158, 0x3159, 0x315a,
        0x315b, 0x315c, 0x315d, 0x315e, 0x315f, 0x3160,
        0x3161, 0x3162, 0x3163
    ])),
    FINAL: list(map(chr, [
        0x3131, 0x3132, 0x3133, 0x3134, 0x3135, 0x3136,
        0x3137, 0x3139, 0x313a, 0x313b, 0x313c, 0x313d,
        0x313e, 0x313f, 0x3140, 0x3141, 0x3142, 0x3144,
        0x3145, 0x3146, 0x3147, 0x3148, 0x314a, 0x314b,
        0x314c, 0x314d, 0x314e
    ]))
}
CHAR_INITIALS = CHAR_LISTS[INITIAL]
CHAR_MEDIALS = CHAR_LISTS[MEDIAL]
CHAR_FINALS = CHAR_LISTS[FINAL]
CHAR_SETS = {k: set(v) for k, v in CHAR_LISTS.items()}
CHARSET = set(itertools.chain(*CHAR_SETS.values()))
CHAR_INDICES = {k: {c: i for i, c in enumerate(v)}
                for k, v in CHAR_LISTS.items()}


def is_hangul_syllable(c):
    return 0xac00 <= ord(c) <= 0xd7a3  # Hangul Syllables


def is_hangul_jamo(c):
    return 0x1100 <= ord(c) <= 0x11ff  # Hangul Jamo


def is_hangul_compat_jamo(c):
    return 0x3130 <= ord(c) <= 0x318f  # Hangul Compatibility Jamo


def is_hangul_jamo_exta(c):
    return 0xa960 <= ord(c) <= 0xa97f  # Hangul Jamo Extended-A


def is_hangul_jamo_extb(c):
    return 0xd7b0 <= ord(c) <= 0xd7ff  # Hangul Jamo Extended-B


def is_hangul(c):
    return (is_hangul_syllable(c) or
            is_hangul_jamo(c) or
            is_hangul_compat_jamo(c) or
            is_hangul_jamo_exta(c) or
            is_hangul_jamo_extb(c))


def is_supported_hangul(c):
    return is_hangul_syllable(c) or is_hangul_compat_jamo(c)


def check_hangul(c, jamo_only=False):
    if not ((jamo_only or is_hangul_compat_jamo(c)) or is_supported_hangul(c)):
        raise ValueError(f"'{c}' is not a supported hangul character. "
                         f"'Hangul Syllables' (0xac00 ~ 0xd7a3) and "
                         f"'Hangul Compatibility Jamos' (0x3130 ~ 0x318f) are "
                         f"supported at the moment.")


def get_jamo_type(c):
    check_hangul(c)
    assert is_hangul_compat_jamo(c), f"not a jamo: {ord(c):x}"
    return sum(t for t, s in CHAR_SETS.items() if c in s)


def split_syllable_char(c):
    """
    Splits a given korean syllable into its components. Each component is
    represented by Unicode in 'Hangul Compatibility Jamo' range.
    Arguments:
        c: A Korean character.
    Returns:
        A triple (initial, medial, final) of Hangul Compatibility Jamos.
        If no jamo corresponds to a position, `None` is returned there.
    Example:
        >>> split_syllable_char("안")
        ("ㅇ", "ㅏ", "ㄴ")
        >>> split_syllable_char("고")
        ("ㄱ", "ㅗ", None)
        >>> split_syllable_char("ㅗ")
        (None, "ㅗ", None)
        >>> split_syllable_char("ㅇ")
        ("ㅇ", None, None)
    """
    check_hangul(c)
    if len(c) != 1:
        raise ValueError("Input string must have exactly one character.")

    init, med, final = None, None, None
    if is_hangul_syllable(c):
        offset = ord(c) - 0xac00
        x = (offset - offset % 28) // 28
        init, med, final = x // 21, x % 21, offset % 28
        if not final:
            final = None
        else:
            final -= 1
    else:
        pos = get_jamo_type(c)
        if pos & INITIAL == INITIAL:
            pos = INITIAL
        elif pos & MEDIAL == MEDIAL:
            pos = MEDIAL
        elif pos & FINAL == FINAL:
            pos = FINAL
        idx = CHAR_INDICES[pos][c]
        if pos == INITIAL:
            init = idx
        elif pos == MEDIAL:
            med = idx
        else:
            final = idx
    return tuple(CHAR_LISTS[pos][idx] if idx is not None else None
                 for pos, idx in
                 zip([INITIAL, MEDIAL, FINAL], [init, med, final]))


def split_syllables(s, ignore_err=True, pad=None):
    """
    Performs syllable-split on a string.
    Arguments:
        s (str): A string (possibly mixed with non-Hangul characters).
        ignore_err (bool): If set False, it ensures that all characters in
            the string are Hangul-splittable and throws a ValueError otherwise.
            (default: True)
        pad (str): Pad empty jamo positions (initial, medial, or final) with
            `pad` character. This is useful for cases where fixed-length
            strings are needed. (default: None)
    Returns:
        Hangul-split string
    Example:
        >>> split_syllables("안녕하세요")
        "ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ"
        >>> split_syllables("안녕하세요~~", ignore_err=False)
        ValueError: encountered an unsupported character: ~ (0x7e)
        >>> split_syllables("안녕하세요ㅛ", pad="x")
        'ㅇㅏㄴㄴㅕㅇㅎㅏxㅅㅔxㅇㅛxxㅛx'
    """

    def try_split(c):
        try:
            return split_syllable_char(c)
        except ValueError:
            if ignore_err:
                return (c,)
            raise ValueError(f"encountered an unsupported character: "
                             f"{c} (0x{ord(c):x})")

    s = map(try_split, s)
    if pad is not None:
        tuples = map(lambda x: tuple(pad if y is None else y for y in x), s)
    else:
        tuples = map(lambda x: filter(None, x), s)
    return "".join(itertools.chain(*tuples))


def join_jamos_char(init, med, final=None):
    """
    Combines jamos into a single syllable.
    Arguments:
        init (str): Initial jao.
        med (str): Medial jamo.
        final (str): Final jamo. If not supplied, the final syllable is made
            without the final. (default: None)
    Returns:
        A Korean syllable.
    """
    chars = (init, med, final)
    for c in filter(None, chars):
        check_hangul(c, jamo_only=True)

    idx = tuple(CHAR_INDICES[pos][c] if c is not None else c
                for pos, c in zip((INITIAL, MEDIAL, FINAL), chars))
    init_idx, med_idx, final_idx = idx
    # final index must be shifted once as
    # final index with 0 points to syllables without final
    final_idx = 0 if final_idx is None else final_idx + 1
    return chr(0xac00 + 28 * 21 * init_idx + 28 * med_idx + final_idx)


def join_jamos(s, ignore_err=True):
    """
    Combines a sequence of jamos to produce a sequence of syllables.
    Arguments:
        s (str): A string (possible mixed with non-jamo characters).
        ignore_err (bool): If set False, it will ensure that all characters
            will be consumed for the making of syllables. It will throw a
            ValueError when it fails to do so. (default: True)
    Returns:
        A string
    Example:
        >>> join_jamos("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안녕하세요"
        >>> join_jamos("ㅇㅏㄴㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ")
        "안ㄴ녕하세요"
        >>> join_jamos()
    """
    last_t = 0
    queue = []
    new_string = ""

    def flush(n=0):
        new_queue = []
        while len(queue) > n:
            new_queue.append(queue.pop())
        if len(new_queue) == 1:
            if not ignore_err:
                raise ValueError(f"invalid jamo character: {new_queue[0]}")
            result = new_queue[0]
        elif len(new_queue) >= 2:
            try:
                result = join_jamos_char(*new_queue)
            except (ValueError, KeyError):
                # Invalid jamo combination
                if not ignore_err:
                    raise ValueError(f"invalid jamo characters: {new_queue}")
                result = "".join(new_queue)
        else:
            result = None
        return result

    for c in s:
        if c not in CHARSET:
            if queue:
                new_c = flush() + c
            else:
                new_c = c
            last_t = 0
        else:
            t = get_jamo_type(c)
            new_c = None
            if t & FINAL == FINAL:
                if not (last_t == MEDIAL):
                    new_c = flush()
            elif t == INITIAL:
                new_c = flush()
            elif t == MEDIAL:
                if last_t & INITIAL == INITIAL:
                    new_c = flush(1)
                else:
                    new_c = flush()
            last_t = t
            queue.insert(0, c)
        if new_c:
            new_string += new_c
    if queue:
        new_string += flush()
    return new_string

def KoG2Padvanced(Sentence):

    # 알고리즘 선행 규칙 추가
    from vall_e.emb.G2P.KoG2P import KoG2P
    import os
    import json
    import re
    import jamo

    import sys, os, re, codecs, string
    import unicodedata
    # import split_syllable_char, split_syllables, join_jamos
    from jamo import h2j, j2hcj
    from konlpy.tag import Kkma
    kkma = Kkma()

    runMorphemeCase = ["의", "히"]

    nInsertionFile = "vall_e/emb/G2P/Dic/nSheetWords.csv"
    nInsertionFr = open(nInsertionFile, 'r')
    nInsertionContent = nInsertionFr.readlines()
    nInsertionFr.close()

    nInsertDic = dict()
    nInsertList = []

    for nInsertContent in nInsertionContent:
        nInsertContentList = nInsertContent.split(",")
        if nInsertContentList[1] == "word":
            pass
        else:
            nInsertDic[nInsertContentList[1]] = nInsertContentList[5].replace("\n", "")
            nInsertList.append(nInsertContentList[1])

    textCSV = Sentence

    sentenceG2P = ""
    words = textCSV.split(" ")
    for word in words:
        # print(word)
        # wordSize = len(word)
        # if wordSize == 1:
        #     sentenceG2P = sentenceG2P + " " + word
        # else:
        word = word.strip()

        for i in range(0, len(runMorphemeCase)):
            if runMorphemeCase[i] in word:
                wordChecked = ""
                kkmaDict = dict(kkma.pos(word))
                if runMorphemeCase[i] == "의":
                    for key, value in kkmaDict.items():
                        # print(key, value)
                        if key == "의" and value == "JKG":
                            wordChecked = wordChecked + "에"
                        else:
                            wordChecked = wordChecked + key
                elif runMorphemeCase[i] == "히":
                    for key, value in kkmaDict.items():
                        jamoInput = split_syllables(key)
                        if "ㅈㅎ" in jamoInput and "VV" == value:  # 맞히다 -> 마치다 / 이히리기의 히가 왔을때만 ㅈㅎ이 ㅊ으로 바뀐다. #낮한때 -> 나탄때 / 낮하늘 -> 나타늘
                            jamoInput = re.sub("ㅈㅎ", "ㅊ", jamoInput)
                        refinedJamo = join_jamos(jamoInput)
                        wordChecked = wordChecked + refinedJamo
                word = wordChecked

        # n-insertion
        for nInsertEach in nInsertList:
            if nInsertEach in word:
                word = word.replace(nInsertEach, nInsertDic[nInsertEach])

        wordJamo = split_syllables(word)
        wordJamo = " " + wordJamo

        # 9. '의' 처리: 한글맞춤법 제9항 ★굳이 안해도 될 것 같은데 선행연구에 '의' 발음 관련 error를 report한 게 있어서 넣었습니다.
        # - 자음으로 시작하는 'ㅢ'는 'ㅣ'로도 발음한다.
        # - 첫음의 '의'를 제외하고 모두 'ㅣ'로도 발음한다.
        # - 부사격 조사 '의' 는 '에' 로도 발음한다: ex) 민주주의의 의의[민주주이에 의이]

        wordJamo = re.sub('(?<![\s])ㅇㅢ', 'ㅇㅣ', wordJamo)
        wordJamo = re.sub('(?<![ㅇ])ㅢ', 'ㅣ', wordJamo)  # 하늬바람 -> 하니바람

        # 논문 에러 -> 중계차 : 중게차
        # 한글 맞춤법 제 8항
        wordJamo = re.sub('(?<=[ㄱㄹㅁㅍㅎ])ㅖ', 'ㅔ', wordJamo)

        # 10. 연음(Liaison)? - 연음이 일어나는 조건:
        # 1 - 1. 자음으로 끝나는 형태소 + 모음으로 시작하는 문법형태소(찾 + 아 -> 찾아[차자], 옷 + 이 -> 옷이[오시]),
        # 1 - 2. 자음으로 끝나는 형태소 + 모음(ㅏ, ㅓ, ㅗ, ㅜ, ㅟ) 으로 시작하는 실질형태소: 자음중화 후 연음(옷안 -> 옫안 -> 오단) (★ 그럼 맛있다[마싣따], 멋있다[머싣따]는...?)
        # 2.(AP 내부의)어절과 어절 사이
        # 3. 단일어 내부

        # # 한사람의????
        #
        # # 연음 형태소 -> 어미와 조사를 봐야한다.
        # # 어미와 조사인 경우에는 자음중화가 일어나지 않고 바로 적용되지만 어미와 조사가 아닌 경우 에는 자음 중화가 일어나고 연음이 이루어져야한다.
        #
        # wordJamo = re.sub('(?<![\sㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖㅇ])ㅇ', '', wordJamo)
        #
        wordJamo = wordJamo.strip()
        wordPhonetic = join_jamos(wordJamo)

        sentenceG2P = sentenceG2P + " " + wordPhonetic

    # print(sentenceG2P)

    fileDir = "vall_e/emb/G2P/Dic/KoG2PDic.txt"
    hangulMo = ["ㅏ", "ㅑ", "ㅓ", "ㅕ", "ㅗ", "ㅛ", "ㅜ", "ㅠ", "ㅡ", "ㅣ", "ㅐ", "ㅘ", "ㅔ", "ㅙ", "ㅚ", "ㅝ", "ㅟ", "ㅞ", "ㅜ", "ㅢ",
                "ㅒ", "ㅖ"]

    fr = open(fileDir, 'r')
    contents = fr.readlines()
    fr.close()
    KoG2PDic = dict()

    for content in contents:
        contentSplit = content.replace("\n", "").strip().split("\t")
        KorSim = contentSplit[0]
        EngSim = contentSplit[1]
        KoG2PDic[EngSim] = KorSim

    totalSentence = ""
    SentenceList = sentenceG2P.strip().split(" ")
    for eachWords in SentenceList:
        KoG2POutcome = str(KoG2P(eachWords))
        KoG2POutcomeList = KoG2POutcome.split(" ")

        KoG2PEngWordList = []
        for KoG2POutcomeEach in KoG2POutcomeList:
            KoG2PEngWordList.append(KoG2PDic[KoG2POutcomeEach])

        HangulString = ""
        for i in range(0, len(KoG2PEngWordList)):
            if i == 0:
                if KoG2PEngWordList[0] in hangulMo:
                    HangulString = HangulString + "ㅇ"
                    HangulString = HangulString + KoG2PEngWordList[i]
                else:
                    HangulString = HangulString + KoG2PEngWordList[i]
            else:
                if i + 1 != len(KoG2PEngWordList):
                    if KoG2PEngWordList[i] in hangulMo and KoG2PEngWordList[i + 1] in hangulMo:
                        HangulString = HangulString + KoG2PEngWordList[i]
                        HangulString = HangulString + "ㅇ"
                    elif KoG2PEngWordList[i] == "ㅇ" and KoG2PEngWordList[i + 1] in hangulMo:
                        HangulString = HangulString + KoG2PEngWordList[i]
                        HangulString = HangulString + "ㅇ"
                    else:
                        HangulString = HangulString + KoG2PEngWordList[i]
                else:
                    HangulString = HangulString + KoG2PEngWordList[i]

        # 2 - 1. ㅎ탈락( / h / -deletion):
        # - ㅎ → Ø / [+son]____[+son] ex) 옳아[올아] / 아홉[아홉~ 아옵], 신혼[신혼~ 시논] / 뚫네[뚤네→뚤레] 뚫는[뚤는→뚤른] -> ㅎ중화 -> 비음화 -> 유음화
        # - ㅎ탈락의 경우 용언에서는 필수적으로 탈락하나 명사에서는 수의적으로 탈락한다

        HangulString = " " + HangulString
        # 초성
        HangulString = re.sub('(?<=[ㅁㄴㅇㄹㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖ])ㅎ(?=[ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖ])', 'ㅇ', HangulString)
        # 종성
        HangulString = re.sub('(?<=[ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖ])ㅎㅇ', 'ㅇ', HangulString)

        # 추가 사항: 자음 위치 동화(Place Assimilation)
        # - 박선우. (2008). 한국어 위치동화의 실험음성학적 분석. 언어연구, 25(2), 45-65.
        # ㄴ, ㄷ → ㅁ, ㅂ / ____{ㅁ, ㅂ, ㅍ, ㅃ} (치경음 → 양순음)
        # 옷보다[옫뽀다]～[옵뽀다] (cf. 오뽀다) / 준비[준비]～[줌비]
        HangulString = re.sub('ㄴ(?=[ㅁㅂㅍㅃ])', 'ㅁ', HangulString)
        HangulString = re.sub('ㄷ(?=[ㅁㅂㅍㅃ])', 'ㅂ', HangulString)

        # ㄴ, ㄷ → ㅇ, ㄱ / ____ {ㄱ, ㅋ, ㄲ}  (치경음 → 연구개음)
        # 숟가락[숟까락]～[숙까락] (cf. 수까락) / 모든 걸[모든걸]～[모등걸]
        HangulString = re.sub('ㄴ(?=[ㄱㅋㄲ])', 'ㅇ', HangulString)
        HangulString = re.sub('ㄷ(?=[ㄱㅋㄲ])', 'ㄱ', HangulString)

        # ㅁ, ㅂ → ㅇ, ㄱ / ____{ㄱ, ㅋ, ㄲ} (양순음 → 연구개음)
        # 숲길[숩낄]～[숙낄] (cf. 수낄) / 짐꾼[짐꾼]～[징꾼]
        HangulString = re.sub('ㅁ(?=[ㄱㅋㄲ])', 'ㅇ', HangulString)
        HangulString = re.sub('ㅂ(?=[ㄱㅋㄲ])', 'ㄱ', HangulString)

        # 연음 형태소 -> 어미와 조사를 봐야한다.
        # 어미와 조사인 경우에는 자음중화가 일어나지 않고 바로 적용되지만 어미와 조사가 아닌 경우 에는 자음 중화가 일어나고 연음이 이루어져야한다.

        # HangulString = HangulString.strip()

        # wordPhonetic = join_jamos(HangulString)

        # kkmaDict_pass = kkma.pos(HangulString)

        # isChecked_EJ = False
        # for i in range(0, len(kkmaDict_pass)):
        #     currentWord_pass = list(kkmaDict_pass[i])
        #     if re.match("^E", currentWord_pass[1]) != None or re.match("^J",currentWord_pass[1]) != None: # 어미나 조사를 포함할 때
        #         jamoInput_pass = split_syllables(currentWord_pass[0]) # 자모로 변환
        #         if jamoInput_pass[0] == "ㅇ": # 현재의 단어가 ㅇ으로 시작될때
        #             isChecked_EJ = True

        # if isChecked_EJ == True:
        #     pass
        # else:
        #     wordJamo_pass = split_syllables(wordPhonetic) #
        #     wordJamo_pass = " " + wordJamo_pass

        #     # 1 - 1. 자음 중화(coda neutralization) (단, ㅎ, ㅀ, ㄶ, ㅌ, 제외)
        #     # - ㅋ, ㄲ → ㄱ / ____ {C,  # }
        #     # - ㅈ, ㅅ, ㅆ, ㅊ → ㄷ / ____ {C,  # }
        #     # - ㅍ → ㅂ / ____ {C,  # }
        #     # - 한국어 음절 구조의 특징은 coda에는 표면형에서 7개의 자음만 올 수 있다는 제약이 있다.그러므로 기저형에 나타나는 coda의 자음들을 7개로 줄여주는 작업이 우선되어야 한다.
        #     # - 'ㅎ', 'ㅌ'을 따로 뺀 이유는 기저형에 'ㅎ', 'ㅌ'이 있는 상태에서 실현되는 음운규칙(ㅎ탈락, 유기음화 / 구개음화)이 존재하기 때문이다.

        #     wordJamo_pass = re.sub('ㅋ(?=[ㅇ])', 'ㄱ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㄲ(?=[ㅇ])', 'ㄱ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㅈ(?=[ㅇ])', 'ㄷ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㅅ(?=[ㅇ])', 'ㄷ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㅆ(?=[ㅇ])', 'ㄷ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㅊ(?=[ㅇ])', 'ㄷ', wordJamo_pass)
        #     wordJamo_pass = re.sub('ㅍ(?=[ㅇ])', 'ㅂ', wordJamo_pass)
        #     wordJamo_pass = wordJamo_pass.strip()
        #     wordPhonetic_pass = join_jamos(wordJamo_pass)
        #     wordPhonetic = wordPhonetic_pass

        # wordJamo_final = split_syllables(wordPhonetic)
        # wordJamo_final = " " + wordJamo_final

        # # 연음 형태소 -> 어미와 조사를 봐야한다.
        # # 어미와 조사인 경우에는 자음중화가 일어나지 않고 바로 적용되지만 어미와 조사가 아닌 경우 에는 자음 중화가 일어나고 연음이 이루어져야한다.

        # HangulString = re.sub('(?<![\sㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖㅇ])ㅇ', '', wordJamo_final)

        HangulString = re.sub('(?<![\sㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅘㅔㅙㅚㅝㅟㅞㅜㅢㅒㅖㅇ])ㅇ', '', HangulString)

        HangulString = HangulString.strip()

        HangulWord = join_jamos(HangulString)
        totalSentence = totalSentence + " " + HangulWord
    totalSentence = totalSentence.strip()

    return totalSentence














