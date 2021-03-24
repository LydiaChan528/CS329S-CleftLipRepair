import os
import pprint
import argparse
import yaml

from model.tools.train import main 

DEFAULT_YAML = "./model/experiments/cleft/face_alignment_cleft_hrnet_w18.yaml"


def parse_config():
    config = argparse.Namespace()
    config.cfg = DEFAULT_YAML
    return config

def edit_yaml():
    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--epochs', help='number of epochs',
                        required=True, type=int)

    parser.add_argument('--pretrained', help='use pretrained weights',
                        action='store_true')

    parser.add_argument('--weights', help='use pretrained weights',
                        required=False, type=str)

    parser.add_argument('--rootdir', help='change root directory',
                        required=False, type=str)

    parser.add_argument('--traindir', help='change train directory',
                        required=False, type=str)

    parser.add_argument('--testdir', help='change test directory',
                        required=False, type=str)

    parser.add_argument('--save_config', help='file to save yaml config in',
                        required=False, type=str)
    
    parser.set_defaults(weights='hrnetv2_pretrained/HR18-COFW.pth')
    
    args = parser.parse_args()

    with open(DEFAULT_YAML) as file:
        params_dict = yaml.load(file, Loader=yaml.FullLoader)

        params_dict['MODEL']['INIT_WEIGHTS'] = args.pretrained
        if args.pretrained:
            params_dict['MODEL']['PRETRAINED'] = args.weights
            
        params_dict['TRAIN']['END_EPOCH'] = args.epochs

        save_file = './model/experiments/cleft/temp.yaml'
        if args.save_config:
            save_file = args.save_config
        with open(save_file , 'w') as f:
            pprint.pprint(params_dict)
            doc = yaml.dump(params_dict, f)
            print(f"Saving new config to {save_file}")
    return args

def run():
    args = edit_yaml()
    main(args)

run()