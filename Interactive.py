import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
from torch import nn
from typing import Dict, List, Optional, Tuple
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm, trange
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers import (
    HfArgumentParser,
    set_seed, BertConfig, BertTokenizer, TrainingArguments,
)

from trainer_ner import Trainer

from model import BertForNER, BertForRelationClassification

from utils_ner import NerDataset, Split, get_labels_ner, get_labels_seq, openue_data_collator

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

def process(text, result):
    index = 0
    start = None
    labels = []
    for w, t in zip(text, result):
        # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "Relation"
        if start is None:
            if t == 'B-SUB' or t == 'B-OBJ':
                start = index
        else:
            # if t == 'I-SUB' or t == 'I-OBJ':
            #     continue
            if t == "O":
                # print(result[start: index])
                labels.append(text[start: index])
                start = None
        index += 1
    # print(labels)
    return labels

def predict(model_seq, model_ner, inputs, training_args, label_map_ner, label_map_seq, texts, tokenizer):

    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(training_args.device)

    inputs_seq = inputs

    with torch.no_grad():
        outputs_seq = model_seq(**inputs_seq)
        relation_output_sigmoid = outputs_seq[0]
        x = torch.max(relation_output_sigmoid, 1)

        x = relation_output_sigmoid > 0.5
        x = x.float()
        y = torch.max(x, 1)[1]  # 先抽单个关系
        z = torch.ones(y.shape).to(training_args.device)  # fault
        predict_relation = y + z  # seq模型预测出来的关系值

        rel_idx = torch.sum(inputs_seq['attention_mask'], 1)

        # 在input_ids拼接上预测出的rel值
        add_rel_idx = rel_idx.long()  # 需要增加rel的位置
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, add_rel_idx)  # fault one-hot向量，用来取、赋值
        t1 = t1.type(torch.uint8)
        inputs_seq['input_ids'][t1] = predict_relation.long()  # 取出rel那一列, 替换

        # 同理，在后面再拼接一个sep符号
        add_rel_idx = add_rel_idx + torch.ones(y.shape).to(training_args.device)  # 后退一个位置
        add_rel_idx = add_rel_idx.long()  # 需要增加rel的位置
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, add_rel_idx)  # one-hot向量，用来取、赋值
        t1 = t1.type(torch.uint8)
        inputs_seq['input_ids'][t1] = 102  # 取出rel那一列, 替换

        # 之后测试还需要修改token_type以及attention_mask的部分
        # attention_mask需要增加两个1（因为增加了rel和sep）
        modify_attntion_idx = rel_idx.long()
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, modify_attntion_idx).type(torch.uint8)
        inputs_seq['attention_mask'][t1] = 1  # 取出那一列, 替换

        modify_attntion_idx = modify_attntion_idx + torch.ones(y.shape).to(training_args.device)
        modify_attntion_idx = modify_attntion_idx.long()
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, modify_attntion_idx).type(torch.uint8)
        inputs_seq['attention_mask'][t1] = 1  # 取出r那一列, 替换

        # token_type同理
        modify_type_idx = rel_idx.long()
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, modify_type_idx).type(torch.uint8)
        inputs_seq['token_type_ids'][t1] = 1  # 取出那一列, 替换

        modify_type_idx = modify_type_idx + torch.ones(y.shape).to(training_args.device)
        modify_type_idx = modify_type_idx.long()
        ones = torch.sparse.torch.eye(inputs_seq['input_ids'].shape[1]).to(training_args.device)
        t1 = ones.index_select(0, modify_type_idx).type(torch.uint8)
        inputs_seq['token_type_ids'][t1] = 1  # 取出那一列, 替换

        inputs_ner = inputs_seq
        # inputs_ner['label_ids_ner'] = _['label_ids_ner']
        outputs_ner = model_ner(**inputs_ner)
        _, results = torch.max(outputs_ner, dim=2)
        results_np = results.cpu().numpy()
        attention_position_np = add_rel_idx.cpu().numpy()

        results_list = results_np.tolist()
        attention_position_list = attention_position_np.tolist()
        predict_relation_list = predict_relation.long().tolist()
        input_ids_list = inputs['input_ids'].tolist()

        processed_results_list = []
        processed_input_ids_list = []
        for idx, result in enumerate(results_list):
            tmp1 = result[0: attention_position_list[idx]-1]
            tmp2 = input_ids_list[idx][0: attention_position_list[idx]-1]
            processed_results_list.append(tmp1)
            processed_input_ids_list.append(tmp2)

        processed_results_list_BIO = []
        for result in processed_results_list:
            processed_results_list_BIO.append([label_map_ner[token] for token in result])

        # 把结果剥离出来
        index = 0
        for ids, BIOS in zip(processed_input_ids_list, processed_results_list_BIO):
            labels = process(ids, BIOS)
            print('\n')
            print(texts[index])
            print(label_map_seq[predict_relation_list[index]])
            for l in labels:
                print(''.join(tokenizer.convert_ids_to_tokens(l)))
            index = index + 1
        print('')

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # 读取ner的label
    # ["O", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ"]
    labels_ner = get_labels_ner()
    label_map_ner: Dict[int, str] = {i: label for i, label in enumerate(labels_ner)}
    num_labels_ner = len(labels_ner)

    # 读取seq的label
    labels_seq = get_labels_seq()
    label_map_seq: Dict[int, str] = {i: label for i, label in enumerate(labels_seq)}
    num_labels_seq = len(labels_seq)

    model_name_or_path = '~/openue_pytorch/output_seq'
    # 读取待训练的seq模型
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels_seq-1,  # fault
        # id2label=label_map_seq,
        label2id={label: i for i, label in enumerate(labels_ner)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model_seq = BertForRelationClassification.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # label_map_seq=label_map_seq,
        # label_map_ner=label_map_ner
    )

    model_name_or_path = '~/openue_pytorch/output_ner'
    # 读取待训练的ner模型
    config = BertConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels_ner,
        id2label=label_map_ner,
        label2id={label: i for i, label in enumerate(labels_ner)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = BertTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model_ner = BertForNER.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # label_map_seq=label_map_seq,
        # label_map_ner=label_map_ner
    )

    model_ner.to(training_args.device)
    model_ner.eval()

    model_seq.to(training_args.device)
    model_seq.eval()

    texts = ['茶树茶网蝽，Stephanitis chinensis Drake，属半翅目网蝽科冠网椿属的一种昆虫',
            '如何演好自己的角色，请读《演员自我修养》《喜剧之王》周星驰崛起于穷困潦倒之中的独门秘笈',
            '《如果能学会不在乎》是由李玲玉演唱的一首歌曲，收录在《大地的母亲》专辑里']

    features = []
    for text in texts:
        inputs_raw = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        features.append(inputs_raw)

    first = features[0]
    batch = {}

    for k, v in first.items():
        if isinstance(v, torch.Tensor):
            batch[k] = torch.stack([f[k] for f in features])
        else:
            batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    predict(model_seq, model_ner, inputs=batch, training_args=training_args,
            label_map_ner=label_map_ner, label_map_seq=label_map_seq,
            texts=texts, tokenizer=tokenizer)

if __name__ == "__main__":
    main()

