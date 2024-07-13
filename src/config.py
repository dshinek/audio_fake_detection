class Config:
    SR = 16000
    N_MFCC = 13
    # Dataset
    ROOT_FOLDER = './'
    TRAIN_VAL_RATE = 0.1
    zero_padding_max = 80000
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 10
    LR = 3e-4
    # Others
    SEED = 42
    #logging
    EXPR_NAME = "dynamic_dataset_including_kaggle_data_with_ast_model_full_finetunning"
    EXPR_DESCRIPTION = "with all data, full finetunning"
    BACKBONE_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
