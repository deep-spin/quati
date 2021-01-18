import argparse
import logging

from quati import config_utils
from quati import opts
from quati import predict
from quati import train

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='quati')
parser.add_argument('task', type=str, choices=['train', 'predict'])
opts.general_opts(parser)
opts.preprocess_opts(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.predict_opts(parser)


if __name__ == '__main__':
    options = parser.parse_args()
    options.output_dir = config_utils.configure_output(options.output_dir)
    config_utils.configure_logger(options.debug, options.output_dir)
    config_utils.configure_seed(options.seed)
    config_utils.configure_device(options.gpu_id)
    logger.info('Output directory is: {}'.format(options.output_dir))

    if options.task == 'train':
        train.run(options)
    elif options.task == 'predict':
        predict.run(options)
