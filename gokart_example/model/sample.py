from logging import getLogger
import luigi
import gokart
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
logger = getLogger(__name__)

class ReadData(gokart.TaskOnKart):
    rerun = True
    data_path = luigi.Parameter()

    def run(self):
        df = pd.read_csv(self.data_path)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userid', 'movieid', 'rating']], reader)
        self.dump(data)


class SplitDataSet(gokart.TaskOnKart):
    rerun = True
    test_size = luigi.FloatParameter(default=0.2)

    def requires(self):
        return ReadData(data_path='data/ratings.csv')
    
    def run(self):
        data = self.load()
        trainset, testset = train_test_split(data, test_size=self.test_size)
        self.dump({'trainset': trainset, 'testset': testset})

class TrainModel(gokart.TaskOnKart):
    rerun=True

    def requires(self):
        return SplitDataSet(test_size=0.2)
    
    def run(self):
        data_dict = self.load() # load data from upstream task
        trainset = data_dict['trainset']

        algo = SVD()
        algo.fit(trainset)
        self.dump(algo)