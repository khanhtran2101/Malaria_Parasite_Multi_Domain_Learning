# with read_base():
#     from mmpretrain.configs._base_.default_runtime import *
#     from mmpretrain.configs._base_.models.resnet18 import *
#     from mmpretrain.configs._base_.schedules.imagenet_bs256 import *


_base_ = [
    #'mmpretrain/configs/resnet/resnet50_8xb32_in1k.py',
    'mmpretrain/configs/_base_/default_runtime.py',
    'mmpretrain/configs/_base_/models/resnet50.py',
    'mmpretrain/configs/_base_/schedules/imagenet_bs256.py'
    #'mmpretrain/configs/_base_/datasets/imagenet_bs32.py',
]

img_norm_cfg = dict(
  mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

#dataset_type = 'CustomDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale = 224),
    dict(type='PackInputs'),
]


# Training dataloader configurations
train_dataloader = dict(
    batch_size = 2,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset', #name of the dataset class,
        data_root='../data/final_malaria_full_class_classification_cropped', # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  # The prefix of file paths in the `ann_file`, relative to the data_root.
        ann_file = "train_annotation.txt",
        with_label=True, # or False for unsupervised tasks
        pipeline=train_pipeline, # The transformations to process the dataset samples.
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

train_dataloader1 = dict(
    batch_size = 2,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset', #name of the dataset class,
        data_root='./data_draft', # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  # The prefix of file paths in the `ann_file`, relative to the data_root.
        with_label=True, # or False for unsupervised tasks
        pipeline=train_pipeline, # The transformations to process the dataset samples.
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

train_cfg = dict(
  #type= "EpochBasedTrainLoop",
  type = "CustomTrainLoop", 
  max_epochs = 5,
  dataloader1 = train_dataloader1,
  val_interval = 1,
  _delete_ = True
)

work_dir='../experiment_result'


model = dict(
    type='CustomClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    #neck=dict(type='Custom_Pooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

# Set val and test components to None to disable validation
val_dataloader = train_dataloader

val_cfg=dict(
    type = "CustomTrainLoop", 
    dataloader1 = train_dataloader1,
    #it must be custom val loop, not custom train loop
    #Override the method if needed
    #validation did not run with the trainloop
)
val_evaluator=dict(type='Accuracy')

# val_cfg = None
# val_evaluator = None

# Optionally set test components to None if they are present in the base config
test_dataloader = None
test_cfg = None
test_evaluator = None

optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)


visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='TensorboardVisBackend')])

# the default value of by_epoch is True
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=10, by_epoch=True))