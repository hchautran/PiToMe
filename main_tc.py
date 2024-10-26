import torch.nn.functional as F
import os

from argparse import ArgumentParser
from algo import (
    PITOME,
    TOME,
    DIFFRATE,
    TOFU,
    DCT,
    NONE,
    MCTF,
    CROSSGET
)
from tasks.tc.engine import Engine, BERT_BASE, DISTILBERT_BASE, BERT_LARGE, ALBERT
    

TASKS = [
    'sst2',
    'imdb',
    'rotten',
    'bbc',
]

model_dict = {
    BERT_BASE: 'BERT-B',
    DISTILBERT_BASE: 'DISTILEDBERT-B',
    BERT_LARGE: 'BERT-L',
    ALBERT: 'ALBERT'
}

if __name__ == "__main__":
    import pathlib
    parser = ArgumentParser()
    parser.add_argument("--task", default="imdb", choices=TASKS,
                        help="choose an LRA dataset from available options")
    parser.add_argument("--algo", default=PITOME, choices=[PITOME, TOME, NONE, TOFU, DCT, DIFFRATE, MCTF, CROSSGET],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--model", default=BERT_BASE, choices=[BERT_BASE, DISTILBERT_BASE, BERT_LARGE, ALBERT],
                        help="choose an LRA dataset from available options")
    parser.add_argument("--ratio", default=0.55, help="remain ratio")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument('--margin',default=None) 
    parser.add_argument('--batch_size', default=8, help='Perform evaluation only')
    args = parser.parse_args()
    batch_size = 16 
    avg_factor = 0.95
    task_name = args.task
    algo = args.algo

    file_name = f'train_tc_{model_dict[args.model]}_{task_name}.csv' if not args.eval else f'eval_tc_{model_dict[args.model]}_{task_name}.csv'
    # file_name = 'ablation_study_alpha.csv' 
    engine = Engine(
        task_name=task_name,
        model_ckt=args.model,
        ratio=float(args.ratio),
        algo=args.algo,
        enable_log=not args.eval,
        trained=args.eval,
        margin=float(args.margin) if args.margin is not None else None,
        alpha=float(args.alpha)
    )
    engine.init_logger()
    if args.eval:
        metrics = engine.evaluate()
    else:
        metrics = engine.train(num_epochs=10)
            
    abs_path =f'{os.getcwd()}/outputs/tc_outputs/'
    path = f'{abs_path}/{file_name}'
    if not os.path.exists(abs_path):
        os.mkdir(abs_path)

    if not pathlib.Path(path).is_file():
        head = "dataset,model,algo,gflops,ratio,acc,eval time,train time,alpha\n"
        with open(path, "a") as myfile:
            myfile.write(head)

    if metrics is not None:
        row = f'{args.task},{args.model},{args.algo},{metrics["gflops"]},{metrics["ratio"]},{metrics["acc"]},{metrics["eval time"]},{metrics["train time"]},{args.alpha}\n'
        with open(path, "a") as myfile:
            myfile.write(row)
                    
                