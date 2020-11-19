# OpenUE Pytorch

用户可通过以下几个简单的步骤实现基于OpenUE的抽取模型训练和部署

1. ***数据预处理***.  下载预训练语言模型 (e.g., [bert-base-chinese](https://github.com/google-research/bert)) 并放置到对应文件夹

2. ***训练分类模型***.
```
python run_seq.py --model_name_or_path $pretrained_model_path --data_dir ～/openue_pytorch/dataset --output_dir ～/openue_pytorch/output_seq --labels dataset/all_50_schemas --save_steps 2000 --num_train_epochs 30 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --do_predict
```
3. ***训练序列标注模型***.
```
python run_ner.py --model_name_or_path $pretrained_model_path --data_dir ~/openue_pytorch/dataset --output_dir /home/bz/openue_pytorch/output_ner --labels dataset/all_50_schemas --save_steps 2000 --num_train_epochs 20 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --do_predict
```
3. ***模型集成与测试***. 
```
python Interactive.py --model_name_or_path $pretrained_model_path --data_dir ~/openue_pytorch/dataset --output_dir ~/openue_pytorch/output --labels dataset/all_50_schemas --save_steps 2000 --num_train_epochs 20 --per_device_train_batch_size 16 --per_device_eval_batch_size 32 --do_predict
```
## 引用

如果您使用或扩展我们的工作，请引用以下文章：

```
@inproceedings{zhang-2020-opennue,
    title = "{O}pe{UE}: An Open Toolkit of Universal Extraction from Text",
    author = "Ningyu Zhang, Shumin Deng, Zhen Bi, Haiyang Yu, Jiacheng Yang, Mosha Chen, Fei Huang, Wei Zhang, Huajun Chen",
    year = "2020",
}
