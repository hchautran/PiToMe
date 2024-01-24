from main_pitome import main as pitome_runner
from main_tome import main as tome_runner
from main_diff import main  as diffrate_runner
from main_pitome import get_args_parser as pitome_args 
from main_tome import get_args_parser as tome_args 
from main_diff import get_args_parser  as diff_args
import argparse

if __name__ == '__main__':
    for algo in [
        'tome',
        'pitome',
        'diffrate',
        'none',
    ]:
        r = 0.9385
        if algo == 'tome':
            parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[tome_args()])
            args = parser.parse_args()
            tome_runner()
        elif algo == 'pitome':
            parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[pitome_args()])
            args = parser.parse_args()
            pitome_runner()
        elif algo == 'diffrate':
            parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[diff_args()])
            args = parser.parse_args()
            diffrate_runner()
        else:
            parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[diff_args()])
            args = parser.parse_args()
            diffrate_runner()
            
            
    
            
