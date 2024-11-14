_base_ = [
    '../mmpretrain/configs/_base_/default_runtime.py',
    '../mmpretrain/configs/_base_/models/resnet50.py',
    '../mmpretrain/configs/_base_/schedules/imagenet_bs256.py'
]

img_norm_cfg = dict(
  mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='PackInputs'),
]


# Training dataloader configurations
train_dataloader = dict(
    batch_size = 2,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root='../data/final_malaria_full_class_classification_cropped', # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "train_annotation.txt",
        with_label=True, 
        pipeline=train_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

val_dataloader = dict(
    batch_size = 2,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root='../data/final_malaria_full_class_classification_cropped', # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "val_annotation.txt",
        with_label=True, 
        pipeline=train_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

test_dataloader = dict(
    batch_size = 2,
    num_workers = 1,
    dataset=dict(
        type='CustomDataset',
        data_root='../data/final_malaria_full_class_classification_cropped', # The common prefix of both `ann_flie` and `data_prefix`.
        data_prefix='',  
        ann_file = "test_annotation.txt",
        with_label=True, 
        pipeline=train_pipeline, 
       ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate')
)

train_cfg = dict(
  #type= "EpochBasedTrainLoop",
  type = "CustomTrainLoop", 
  max_epochs = 1,
  dataloader = train_dataloader,
  dataloader1 = train_dataloader,
  _delete_ = True
)

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=6,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

val_evaluator = None


optimizer = dict(type='Adam', lr=0.00001, weight_decay=0.0001)


visualizer = dict(
    type='Visualizer', 
    vis_backends=[dict(type='TensorboardVisBackend')])

checkpoint=dict(type='CheckpointHook', interval=0)