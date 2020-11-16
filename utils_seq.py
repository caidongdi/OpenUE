# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
import json
import jsonlines
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import numpy as np

from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


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

    # guid: str
    # words: List[str]
    # labels: Optional[List[str]]

    id: int
    text: str
    # labels: Optional[List[str]]
    # subject: str
    # predicate: str
    # object: str
    triples: List[List]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "valid"
    test = "test"


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class OpenueDataset(Dataset):
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
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir, "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            # lock_path = cached_features_file + ".lock"
            # with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    labels,
                    max_seq_length,
                    tokenizer,
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

            print("feature prepared.")

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.json")
    guid_index = 1
    examples = []
    # with open(file_path, encoding="utf-8") as f:
    #     words = []
    #     labels = []
    #     for line in f:
    #         if line.startswith("-DOCSTART-") or line == "" or line == "\n":
    #             if words:
    #                 examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
    #                 guid_index += 1
    #                 words = []
    #                 labels = []
    #         else:
    #             splits = line.split(" ")
    #             words.append(splits[0])
    #             if len(splits) > 1:
    #                 labels.append(splits[-1].replace("\n", ""))
    #             else:
    #                 # Examples could have no label for mode = "test"
    #                 labels.append("O")
    #     if words:
    #         examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))

    with open(file_path, "r+", encoding="utf8") as f:
        counter = 0
        for item in jsonlines.Reader(f):
            # print(item)
            text = item['text']
            if len(item['spo_list']) != 0:
                triples = []

                for triple in item['spo_list']:
                    subject = triple['subject']
                    predicate = triple['predicate']
                    object = triple['object']
                    triples.append([subject, predicate, object])

                counter = counter + 1
                examples.append(InputExample(id=counter, text=text, triples=triples))

    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
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

    label_map = {label: i for i, label in enumerate(label_list)}

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        # 用bert分词，转换为token
        text = example.text

        # label_id_ = np.eye(len(label_list))[label_id]

        label_id_ = np.zeros(len(label_list))

        for item in example.triples:
            subject = item[0]
            predicate = item[1]
            object = item[2]

            label_id_tmp = label_map[predicate]
            label_id_[label_id_tmp] = 1

        # tokens = tokenizer.tokenize(text)

        # 转换为id，加上cls以及seq等
        # {"input_ids":[], "token_type_ids":[], "attention_mask":[]}
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        features.append(
            InputFeatures(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                label_ids=label_id_.tolist()
                # label_ids=label_id
            )
        )

    return features


def get_labels(path: str) -> List[str]:
    # if path:
    #     with open(path, "r") as f:
    #         labels = f.read().splitlines()
    #     if "O" not in labels:
    #         labels = ["O"] + labels
    #     return labels
    # else:
    #     return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    # dataset/all_50_schemas
    class_label = ['丈夫', '上映时间', '专业代码', '主持人', '主演', '主角', '人口数量', '作曲', '作者', '作词', '修业年限', '出品公司', '出版社', '出生地', '出生日期','创始人', '制片人', '占地面积', '号', '嘉宾', '国籍', '妻子', '字', '官方语言', '导演', '总部地点', '成立日期', '所在城市', '所属专辑', '改编自', '朝代', '歌手', '母亲', '毕业院校', '民族', '气候', '注册资本', '海拔', '父亲', '目', '祖籍', '简称', '编剧', '董事长', '身高', '连载网站','邮政编码', '面积', '首都']
    schema = {
    '父亲': [('人物', '人物')],
    '妻子': [('人物', '人物')],
    '母亲': [('人物', '人物')],
    '丈夫': [('人物', '人物')],
    '祖籍': [('地点', '人物')],
    '总部地点': [('地点', '企业')],
    '出生地': [('地点', '人物')],
    '目': [('目', '生物')],
    '面积': [('Number', '行政区')],
    '简称': [('Text', '机构')],
    '上映时间': [('Date', '影视作品')],
    '所属专辑': [('音乐专辑', '歌曲')],
    '注册资本': [('Number', '企业')],
    '首都': [('城市', '国家')],
    '导演': [('人物', '影视作品')],
    '字': [('Text', '历史人物')],
    '身高': [('Number', '人物')],
    '出品公司': [('企业', '影视作品')],
    '修业年限': [('Number', '学科专业')],
    '出生日期': [('Date', '人物')],
    '制片人': [('人物', '影视作品')],
    '编剧': [('人物', '影视作品')],
    '国籍': [('国家', '人物')],
    '海拔': [('Number', '地点')],
    '连载网站': [('网站', '网络小说')],
    '朝代': [('Text', '历史人物')],
    '民族': [('Text', '人物')],
    '号': [('Text', '历史人物')],
    '出版社': [('出版社', '书籍')],
    '主持人': [('人物', '电视综艺')],
    '专业代码': [('Text', '学科专业')],
    '歌手': [('人物', '歌曲')],
    '作词': [('人物', '歌曲')],
    '主角': [('人物', '网络小说')],
    '董事长': [('人物', '企业')],
    '成立日期': [('Date', '机构'), ('Date', '企业')],
    '毕业院校': [('学校', '人物')],
    '占地面积': [('Number', '机构')],
    '官方语言': [('语言', '国家')],
    '邮政编码': [('Text', '行政区')],
    '人口数量': [('Number', '行政区')],
    '所在城市': [('城市', '景点')],
    '作者': [('人物', '图书作品')],
    '作曲': [('人物', '歌曲')],
    '气候': [('气候', '行政区')],
    '嘉宾': [('人物', '电视综艺')],
    '主演': [('人物', '影视作品')],
    '改编自': [('作品', '影视作品')],
    '创始人': [('人物', '企业')]}

    return class_label