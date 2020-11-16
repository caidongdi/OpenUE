""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from typing import Dict

import numpy as np

import jsonlines

from transformers import PreTrainedTokenizer, is_torch_available, BatchEncoding, AutoConfig, AutoTokenizer
from transformers.data.data_collator import InputDataClass

from tokenizers import ByteLevelBPETokenizer, SentencePieceBPETokenizer, BertWordPieceTokenizer

logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    text_id: str
    words: List[str]
    relation: List[str]
    subject: List[str]
    object_: List[str]
    # label_ner: List[str]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None

    input_ids_raw: List[int] = None
    attention_mask_raw: Optional[List[int]] = None
    token_type_ids_raw: Optional[List[int]] = None

    label_ids_seq: Optional[List[int]] = None
    label_ids_ner: Optional[List[int]] = None

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "valid"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class NerDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels_seq: List[str],
            labels_ner: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            # Load data features from cache or dataset file
            cached_examples_file = os.path.join(
                data_dir, "cached_ner_.examples".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )
            cached_features_file = os.path.join(
                data_dir, "cached_ner_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            # lock_path = cached_features_file + ".lock"
            # with FileLock(lock_path):
            examples = None

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:

                if os.path.exists(cached_examples_file) and not overwrite_cache:
                    logger.info(f"Loading example from dataset file at {data_dir}")
                    examples = torch.load(cached_examples_file)
                else:
                    logger.info(f"Creating example from cached file {cached_examples_file}")
                    examples = read_examples_from_file(data_dir, mode)
                    torch.save(examples, cached_examples_file)
                # TODO clean up all this to leverage built-in features of tokenizers
                logger.info(f"Creating features from dataset file at {data_dir}")
                self.features = convert_examples_to_features(
                    examples,
                    # labels,
                    labels_seq=labels_seq,
                    labels_ner=labels_ner,
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                    cls_token_at_end=bool(model_type in ["xlnet"]),
                    # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                    sep_token=tokenizer.sep_token,
                    sep_token_extra=False,
                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")

    examples = []
    # text_rel_label = {}

    with open(file_path, "r+", encoding="utf8") as f:

        text_id = 0

        for item in jsonlines.Reader(f):
            # print(item)
            text = item['text']
            if len(item['spo_list']) != 0:
                for triple in item['spo_list']:
                    subject = triple['subject']
                    predicate = triple['predicate']
                    object = triple['object']

                    # B-SUB I-SUB / B-OBJ I-OBJ
                    # text[21: 21+len(subject)]

                    # lable_NER = ['O' for i in range(len(text))]
                    #
                    # # 标注subject
                    # idx_start = text.find(subject)
                    # idx_end = idx_start + len(subject) - 1
                    # flag = True
                    # while idx_start <= idx_end:
                    #     if flag:
                    #         lable_NER[idx_start] = 'B-SUB'
                    #         flag = False
                    #     else:
                    #         lable_NER[idx_start] = 'I-SUB'
                    #     idx_start = idx_start + 1
                    #
                    # # 标注object
                    # idx_start = text.find(object)
                    # idx_end = idx_start + len(object) - 1
                    # flag = True
                    # while idx_start <= idx_end:
                    #     if flag:
                    #         lable_NER[idx_start] = 'B-OBJ'
                    #         flag = False
                    #     else:
                    #         lable_NER[idx_start] = 'I-OBJ'
                    #     idx_start = idx_start + 1

                    # 文本 + 特定关系 + 对应三元组 + 对应label
                    examples.append(InputExample(text_id=text_id, words=text, relation=predicate,
                                                 subject=subject, object_=object))
                    text_id = text_id + 1

                    # text_rel_label[text_id][predicate] = lable_NER

    return examples

def convert_examples_to_features(
    examples: List[InputExample],
    # label_list: List[str],
    labels_seq: List[str],
    labels_ner: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map_seq = {label: i for i, label in enumerate(labels_seq)}
    label_map_ner = {label: i for i, label in enumerate(labels_ner)}

    features = []
    # counter = 0

    def find_word_in_texts(word_ids, texts_ids):
        length = len(word_ids)
        for i, W in enumerate(texts_ids):
            if texts_ids[i] == word_ids[0]:
                if texts_ids[i: i + length] == word_ids:
                    return i, i + length
        return None, None

    for (ex_index, example) in enumerate(examples):
        # 用bert分词，转换为token
        # text = example.text
        if ex_index > 10000:
            break
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        text = example.words
        relation = example.relation
        subject = example.subject
        object_ = example.object_

        # cls w1 w2 .. sep w3 w4 sep 000000000
        # token_type
        # 000000000000000 1111111111
        # 转换为id，加上cls以及seq等
        # {"input_ids":[], "token_type_ids":[], "attention_mask":[]}
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            # max_length=None,
            # padding="max_length",
            # truncation=True,
            return_overflowing_tokens=True,
        )

        inputs_raw = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        inputs['token_type_ids'] = tokenizer.create_token_type_ids_from_sequences(inputs['input_ids'][1:-1],
                                                                                   [label_map_seq[relation]])
        inputs['input_ids'] = inputs['input_ids'] + [label_map_seq[relation], tokenizer.sep_token_id]
        inputs['attention_mask'] = inputs['attention_mask'] + [1, 1]

        # 添加split_text文本的标签
        # B-SUB I-SUB / B-OBJ I-OBJ
        split_text_ids = inputs['input_ids']
        # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]
        # 默认所有位置都为'O'
        lable_ner = ['O' for i in range(len(split_text_ids))]

        # 标注subject
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        [start_idx, end_idx] = find_word_in_texts(subject_ids, split_text_ids)
        if start_idx is None:
            logger.info('语料有问题(subject)！%d', ex_index)
            continue
        lable_ner[start_idx: end_idx] = ['I-SUB' for i in range(len(subject_ids))]
        lable_ner[start_idx] = 'B-SUB'

        # 标注object
        object_ids = tokenizer.encode(object_, add_special_tokens=False)
        [start_idx, end_idx] = find_word_in_texts(object_ids, split_text_ids)
        if start_idx is None:
            logger.info('语料有问题(object)！%d', ex_index)
            continue
        lable_ner[start_idx: end_idx] = ['I-OBJ' for i in range(len(object_ids))]
        lable_ner[start_idx] = 'B-OBJ'

        # 标注最后三个字符串，SEP、Relation、SEP
        lable_ner[-1] = 'O'
        lable_ner[-2] = 'Relation'
        lable_ner[-3] = 'O'

        assert len(lable_ner) == len(inputs['input_ids']) == len(inputs['token_type_ids']) ==\
                    len(inputs['attention_mask'])

        # 关系抽取标签
        label_id_seq = label_map_seq[relation]
        # label_id_seq_special = [0 for i in range(len(labels_seq))]
        # label_id_seq_special[label_id_seq] = 1

        # NER标签转换
        label_id_ner = [label_map_ner[i] for i in lable_ner]

        # 这里需要验证一下，因为每一个输出字符都需要预测类别
        # 将label_id_ner转换为标签的格式
        # max_seq_length * num_ner_labels
        # tmp = np.zeros([max_seq_length, len(labels_ner)])
        # tmp = tmp.tolist()
        #
        # for index, token in enumerate(label_id_ner):
        #     tmp[index][label_id_ner[index]] = 1
        #
        # label_id_ner = tmp

        features.append(
            InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],

                input_ids_raw=inputs_raw["input_ids"],
                attention_mask_raw=inputs_raw["attention_mask"],
                token_type_ids_raw=inputs_raw["token_type_ids"],

                label_ids_ner=label_id_ner,
                label_ids_seq=label_id_seq
            )
        )

    # print('超过max_length的句子比例是', str(counter/len(examples)))
    return features


def get_labels_ner() -> List[str]:
    # if path:
    #     with open(path, "r") as f:
    #         labels = f.read().splitlines()
    #     if "O" not in labels:
    #         labels = ["O"] + labels
    #     return labels
    # else:
    #     return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    return ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"]

def get_labels_seq() -> List[str]:
    class_label = ['Empty', '丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期','创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站','邮政编码', '面积', '首都']
    return class_label

def openue_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    # 读取ner的label
    labels_ner = get_labels_ner()
    label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(labels_ner)}
    num_labels_ner = len(labels_ner)

    labels_seq = get_labels_seq()
    label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(labels_seq)}
    num_labels_seq = len(labels_seq)

    max_length = [len(f.input_ids) for f in features]
    max_length = max(max_length)

    for f in features:
        length = len(f.input_ids)
        distance = max_length - length
        add_zero = [0 for i in range(distance)]
        add_special = [-100 for i in range(distance)]
        f.input_ids = f.input_ids + add_zero  # 补0
        f.attention_mask = f.attention_mask + add_zero  # 补0
        f.token_type_ids = f.token_type_ids + add_zero  # 补0
        f.label_ids_ner = f.label_ids_ner + add_special  # 补-100,这里仅仅为了补齐的最长长度, loss计算中有mask存在会被忽略
        # 处理seq label的格式
        # t1 = np.zeros(num_labels_seq)
        # t1[f.label_ids_seq] = 1
        # f.label_ids_seq = t1.tolist()
        # 处理ner label的格式
        # for i, token_label in enumerate(f.label_ids_ner):
        #     t1 = np.zeros(num_labels_ner)
        #     t1[token_label] = 1
        #     f.label_ids_ner[i] = t1.tolist()

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    label = first["label_ids_seq"].item() if isinstance(first["label_ids_seq"], torch.Tensor) else first["label_ids_seq"]
    dtype = torch.long if isinstance(label, int) else torch.long
    batch["label_ids_seq"] = torch.tensor([f["label_ids_seq"] for f in features], dtype=dtype)

    label = first["label_ids_ner"].item() if isinstance(first["label_ids_ner"], torch.Tensor) else first["label_ids_ner"]
    dtype = torch.long if isinstance(label, int) else torch.long
    batch["label_ids_ner"] = torch.tensor([f["label_ids_ner"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label_ids_seq", "label_ids_ner") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch