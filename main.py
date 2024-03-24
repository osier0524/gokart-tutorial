import luigi
import numpy as np
import gokart

import gokart_example

if __name__ == '__main__':
    gokart.add_config('./conf/param.ini')
    gokart.run([
        'TrainModel',
        '--local-scheduler',
    ])
