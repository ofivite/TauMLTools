import os
import json
import wandb
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

if os.path.exists("wanb_keys.json"):
   os.environ["WANDB_API_KEY"] = json.loads("wanb_keys.json")["work_account"]
# os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_START_METHOD"]="thread"

@hydra.main(config_path=".", config_name="log_roc_cfg")
def main(cfg: DictConfig) -> None:
    # read input json file
    path_to_json = to_absolute_path(f'{cfg.input_path}/tau_vs_{cfg.vs_type}_{cfg.dataset}.json')
    try:
        with open(path_to_json, 'r') as f:
            roc_cfg = json.load(f)
    except FileNotFoundError as e:
        print(f'\n--- {path_to_json} doesn\'t exist, skipping it\n')
        return

    # retrieve the discriminator for a given name and pt bin
    found_discriminator = False
    for discriminator_data in roc_cfg[cfg.pt_bin]['discriminators']:
        if discriminator_data['name'] != cfg.discriminator:
            continue
        else:
            found_discriminator = True
            break
    if not found_discriminator: return

    # initialise wandb run and log its parameters
    params = {'vs_type': cfg.vs_type,
              'dataset': cfg.dataset,
              'pt_min': roc_cfg[cfg.pt_bin]['pt_min'],
              'pt_max': roc_cfg[cfg.pt_bin]['pt_max'],
              'discriminator': cfg.discriminator,
              'auc_score': discriminator_data['auc_score'],
              'period': roc_cfg[cfg.pt_bin]['period'],
              }
    wandb.init(name=f"{params['discriminator']}, {params['vs_type']}, {params['dataset']}, {params['pt_min']}",
               config=params, project='deeptau',reinit=True)

    # plot roc curve from FPR and TPR
    roc_rates = list(zip(discriminator_data['false_positive_rate'], discriminator_data['true_positive_rate']))
    table = wandb.Table(columns=["fpr", "tpr"], data=roc_rates)
    roc_curve = wandb.plot_table(
        "wandb/area-under-curve/v0",
        table,
        {"x": "fpr", "y": "tpr"},
        {
            "title": "ROC",
            "x-axis-title": "False positive rate",
            "y-axis-title": "True positive rate",
        },
    )
    wandb.log({"roc_curve" : roc_curve})
    wandb.finish()
    
if __name__ == '__main__':
    main()
