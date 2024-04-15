import torch.nn.functional as F

from tc.lra_datasets import (ListOpsDataset, Cifar10Dataset, ImdbDataset)
from argparse import ArgumentParser
from accelerate import Accelerator
from dotenv import load_dotenv
from algo import (
    PITOME,
    TOME,
    DIFFRATE,
    TOFU,
    DCT,
    NONE
)
from tc.engine import Engine, BERT_BASE, DISTILBERT_BASE, BERT_LARGE


# consts
OUTPUT_DIR = "output_dir/"
deepspeed_json = "ds_config.json"

TASKS = [
    'sst2',
    'imdb',
    'rotten',
    'bbc',
]

model_dict = {
    BERT_BASE: 'BERT-B',
    DISTILBERT_BASE: 'DISTILEDBERT-B',
    BERT_LARGE: 'BERT-L'
}

if __name__ == "__main__":
    import pathlib
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS,
                        help="choose an LRA dataset from available options")
    parser.add_argument("--algo", default=PITOME, choices=[PITOME, TOME, NONE, TOFU, DCT, DIFFRATE],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--model", default=BERT_BASE, choices=[BERT_BASE, DISTILBERT_BASE, BERT_LARGE],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--ratio", default=0.55, help="remain ratio")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--batch_size', default=8, help='Perform evaluation only')
    args = parser.parse_args()
    batch_size = 4 
    avg_factor = 0.95
    task_name = args.task
    algo = args.algo

    file_name = f'train_tc_{model_dict[args.model]}_{task_name}.csv' if not args.eval else f'eval_tc_{model_dict[args.model]}_{task_name}.csv'
    print(file_name)
    engine = Engine(
        task_name=task_name,
        model_ckt=args.model,
        ratio=float(args.ratio),
        algo=args.algo,
        enable_log=not args.eval,
        trained=args.eval
    )
    engine.init_logger()
    if args.eval:
        metrics = engine.evaluate()
    else:
        metrics = engine.train(num_epochs=10)
            
    abs_path ='/home/caduser/HDD/vit_token_compress/PiToMe/'
    path = f'{abs_path}/{file_name}'
    if not pathlib.Path(path).is_file():
        head = "dataset,model,algo,gflops,ratio,acc,eval time,train time\n"
        with open(file_name, "a") as myfile:
            myfile.write(head)

    if metrics is not None:
        row = f'{args.task}, {args.model}, {args.algo}, {metrics["gflops"]}, {metrics["ratio"]}, {metrics["acc"]}, {metrics["eval time"]}, {metrics["train time"]}\n'
        with open(file_name, "a") as myfile:
            myfile.write(row)
                    
                