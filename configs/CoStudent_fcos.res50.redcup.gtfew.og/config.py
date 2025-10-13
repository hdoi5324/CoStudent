import os.path as osp

from cvpods.configs.fcos_config import FCOSConfig
from cvpods.configs.coco import COCOMutiBranch  # noqa

_config_dict = dict(
    SEED=2234575,
    OUTPUT_DIR="outputs/loose/redcup_og_costudent",
    MODEL=dict(
        WEIGHTS="./models/R-50Imagenet_backboneonly.pkl",
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            NUM_CLASSES=1,
            CENTERNESS_ON_REG=True,
            NORM_REG_TARGETS=True,
            NMS_THRESH_TEST=0.6,
            BBOX_REG_WEIGHTS=(1.0, 1.0, 1.0, 1.0),
            FOCAL_LOSS_GAMMA=2.0,
            FOCAL_LOSS_ALPHA=0.25,
            IOU_LOSS_TYPE="giou",
            CENTER_SAMPLING_RADIUS=1.5,
            OBJECT_SIZES_OF_INTEREST=[
                [-1, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, float("inf")],
            ],
            PSEUDO_SCORE_THRES=0.5,
            MATCHING_IOU_THRES=0.4,
            WITH_TEACHER=True,
        ),
    ),
    DATASETS=dict(
        CO_MINING=True,
        TRAIN=("coco_squidle_redcup_train_sparse",),
        TEST=("coco_squidle_redcup_test",),
    ),
    TRAINER=dict(
        NAME="MultiBranchRunner",
        FP16=dict(
            ENABLED=False,
            # options: [APEX, PyTorch]
            TYPE="APEX",
            # OPTS: kwargs for each option
            OPTS=dict(
                OPT_LEVEL="O1",
            ),
        ),
        WITH_TEACHER=True
    ),
    
    SOLVER=dict(
            CHECKPOINT_PERIOD=250,
            LR_SCHEDULER=dict(
                MAX_ITER=3000, # divided by 10 for smaller dataset
                STEPS=(2000, 2500), # divided by 10 for smaller dataset
            ),
            OPTIMIZER=dict(
                BASE_LR=0.0125, # learning rate in original config is used for 8 GPUs 16 total batch;
            ),
            IMS_PER_DEVICE=4, # Creates a batch of 16 for 2 GPUs
    ),
    TEST=dict(
        EVAL_PERIOD=250,
        DETECTIONS_PER_IMAGE=100,
    ),
    DATALOADER=dict(
        # Number of data loading threads
        NUM_WORKERS=8),
    INPUT=dict(
        AUG=dict(
            TRAIN_PIPELINES=[
                
                [ dict(type = "OrderList", 
                            transforms = [dict(type ="RandResizeShortestEdge",
                                        short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                        ],
                            record = True)
                ],

                [ dict(type = "OrderList", transforms = 
                                                [dict(type ="RandResizeShortestEdge",
                                                     short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                                 dict(type ="RandFlip",prob=0.5),
                                                 dict(type="ColorJiter")],
                                                 record = True
                                                 )
                ],


                [ dict(type = "OrderList", transforms = 
                                                [dict(type ="RandResizeShortestEdge",
                                                     short_edge_length=(800,), max_size=1333, sample_style="choice"),
                                                 dict(type ="RandFlip",prob=0.5),
                                                 dict(
                                                    type="ShuffleList",
                                                    transforms=[
                                                        dict(
                                                            type="OneOf",
                                                            transforms=[
                                                                dict(type = "RandAutoAugment",name=k,magnitude=10,)
                                                                        for k in [
                                                                            "Identity",
                                                                            "AutoContrast",
                                                                            "Equalize",
                                                                            "Solarize",
                                                                            "Color",
                                                                            "Contrast",
                                                                            "Brightness",
                                                                            "Sharpness",
                                                                            "PosterizeResearch",
                                                                        ]
                                                            ],
                                                        ),
                                                        dict(
                                                            type="OneOf",
                                                            transforms=[
                                                                dict(type="RandTranslate", x=(-0.1, 0.1)),
                                                                dict(type="RandShear", x=(-30, 30)),
                                                                dict(type="RandShear", y=(-30, 30)),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                    dict(
                                                        type="RandErase",
                                                        n_iterations=(1, 5),
                                                        size=[0, 0.2],
                                                        squared=True,
                                                    ),
                                                 ],
                                                 record = True
                                )
                ],
            ],
            TEST_PIPELINES=[
                ("ResizeShortestEdge",
                 dict(short_edge_length=(1024, 1024), max_size=2048, sample_style='choice')), # reduces short size to 1024 keeping long edge at or below 2048
            ],
        ),
    ),
)


class CustomFCOSConfig(FCOSConfig):
    def __init__(self):
        super(CustomFCOSConfig, self).__init__()
        self._register_configuration(_config_dict)


config = CustomFCOSConfig()
