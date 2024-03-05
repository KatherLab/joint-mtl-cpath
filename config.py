import torch
import json

class TrainConfig:
    # Targets and data
    target_label_class="isMSIH"
    label_mapping = {'MSS': 0, 'MSI-H': 1}
    target_label_regr="Intratumor Heterogeneity"
    clini_table="/path/to/clini_table.xlsx"
    slide_table="/path/to/slide_table.csv"
    feature_dir="/path/to/feature/vectors/STAMP_raw_xiyuewang-ctranspath-7c998680"

    # Training-related configuration
    baseline=False # If True, the baseline model is trained
    task_type= "classification" # Only valid for baseline. "joint", "regression", "classification"
    weight="autol" # task weighting methods: equal, dwa, uncert, autol. Not valid with baseline.
    grad_method="cagrad" # gradient weighting methods: graddrop, pcgrad, cagrad. Not valid with baseline.
    early_stopping=7
    k_folds = 5
    num_epochs = 32
    batch_size = 1
    instances_per_bag = None
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_type= weight + "_" + grad_method # string for saving model

    # Model-related configuration
    learning_rate = 0.0001
    num_encoder_heads = 6
    num_decoder_heads = 6
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_model = 384
    dim_feedforward = 768
    positional_encoding = False
    model_name = loss_type+"_"+target_label_class+"_"+target_label_regr+"_"+task_type+"-mtl-cpath"  # Name for saving the model
    
    # For logging reasons
    if baseline:
        weight=None
        grad_method=None
        model_name = "baseline_"+target_label_class+"_"+target_label_regr+"_"+task_type+"-mtl-cpath"  # Name for saving the model
        loss_type="baseline"


    def save_to_json(self, file_path):
        config_dict = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
    
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

    pass


class TestConfig:
    #targets
    target_label_class="isMSIH"
    label_mapping = {'MSS': 0, 'MSI-H': 1}
    target_label_regr="Intratumor Heterogeneity"
    task_type= "joint" #"joint", "regression", "classification"
    deploy_folder="autol_cagrad_isMSIH_Intratumor Heterogeneity_joint-mtl-cpath_xxxxx"
    model_path=f"/path/to/{deploy_folder}"
    clini_table="/path/to/external/cohort/clini_table.xlsx"
    slide_table="/path/to/external/cohort/slide_table.csv"
    feature_dir="/path/to/external/cohort/feature/vectors/STAMP_raw_xiyuewang-ctranspath-7c998680"
    batch_size = 1
    instances_per_bag = None
    num_workers = 4

    # Model-related configuration
    loss_type=deploy_folder.split('_')[:2]
    dummy_regr=True
    learning_rate = 0.0001
    num_encoder_heads = 6
    num_decoder_heads = 6
    num_encoder_layers = 2
    num_decoder_layers = 2
    d_model = 384
    dim_feedforward = 768
    positional_encoding = False
    model_name = f"{deploy_folder}_deploy_CPTAC"  # Choose a suitable name for your model

    # Training-related configuration
    early_stopping=7
    k_folds = 5
    num_epochs = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def save_to_json(self, file_path):
        config_dict = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
    
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

    pass
