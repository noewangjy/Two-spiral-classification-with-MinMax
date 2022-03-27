import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
from multiprocessing import Pool
from data_utils import TwoSpiralDataSet 
import time

from modeling import BaseModel
from optimizer import Optimizer


class Trainer(object):
    def __init__(self):
        self._loss = None
        self._total_cnt = None
        self._derivative = None
        self._positive_cnt = None

    def _initialize(self):
        self._loss = 0.0
        self._total_cnt = 0
        self._positive_cnt = 0
    
    def load_model(self, model_path="."):
        if not os.path.isfile(model_path):
            raise ValueError(f"Model path '{model_path}' is NOT a file!")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        return model
        
    def train(self, 
        model : BaseModel, 
        optimizer: Optimizer, 
        X_train: np.ndarray, 
        Y_train: np.array, 
        epochs: int=1, 
        verbose: bool=True,
        early_stop: bool=True,
        ):

        self._initialize()
        assert epochs >= 1
        if verbose:
            print(f"Total epochs: {epochs}, Total data: {X_train.shape[0]}")
            with tqdm(total=epochs) as pbar:
                for epoch in range(epochs):
                    self._loss = 0
                    self._total_cnt = 0
                    for i in range(X_train.shape[0]):
                        X = np.expand_dims(X_train[i, :], axis=0)
                        Y = np.expand_dims(Y_train[i], axis=0)
                        output = model.forward(X)
                        self._loss += 0.5 * np.sum((Y - output)**2)
                        self._total_cnt += 1
                        self._derivative = -(Y - output)
                        model.backward(self._derivative)
                        optimizer.optimize(model)
                    if epoch % 500 == 0:
                        self.polt_decision_boundary(model, index=str(epoch))
                    if early_stop and (self._loss/self._total_cnt < 1e-2):
                        print(f"Early stopped at epoch {epoch}, loss = {self._loss/self._total_cnt}")
                        break

                    pbar.set_description("Loss: {:.6f}".format(self._loss/self._total_cnt))
                    pbar.update()


        else:
            for epoch in range(epochs):
                for i in range(X_train.shape[0]):
                    X = np.expand_dims(X_train[i, :], axis=0)
                    Y = np.expand_dims(Y_train[i], axis=0)
                    output = model.forward(X)
                    self._loss += 0.5 * np.sum((Y - output)**2)
                    self._total_cnt += 1
                    self._derivative = -(Y - output)
                    model.backward(self._derivative)
                    optimizer.optimize(model)
                if early_stop and (self._loss/self._total_cnt < 1e-2):
                    break

        return model

    def test(self, model: BaseModel, X_test: np.ndarray, Y_test: np.array):
        self._initialize()
        print(f"Test data size: {len(Y_test)}")
        for i in tqdm(range(len(Y_test))):
            X = np.expand_dims(X_test[i, :], axis=0)
            Y = Y_test[i]
            prediction = model.forward(X)
            self._loss += 0.5 * (Y - prediction)**2
            self._total_cnt += 1
            if (prediction > 0.5 and Y) or (prediction <=0.5 and Y==0):
                self._positive_cnt += 1
        
        print(f"Loss: {self._loss/self._total_cnt}, Accuracy: {self._positive_cnt/self._total_cnt}.")
    
    def predict(self, model: BaseModel, X_test: np.ndarray):
        # self._initialize()
        predictions = []
        for i in range(X_test.shape[0]):
            X = np.expand_dims(X_test[i, :], axis=0)
            predictions.append(model.forward(X))
        predictions = np.array(predictions).reshape(-1, 1)

        return predictions


    def save_model(self, model: BaseModel, save_path: str=".", index: str="0"):
        path = os.path.join(save_path, index + "_model.pkl")
        if os.path.exists(save_path):
            print(f"Saving model to path: {path}")
        else:
            os.makedirs(save_path)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Successfully saved model to {path}")

    def visual_test(self, model: BaseModel, X_test: np.ndarray, Y_test: np.ndarray, save_path: str="figures/SingleModel", index: str="0"):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self._initialize()
        predictions = []
        results = []
        for i in tqdm(range(len(Y_test))):
            X = np.expand_dims(X_test[i, :], axis=0)
            Y = Y_test[i]
            prediction = np.round(model.forward(X))
            predictions.append(prediction)
            if prediction == Y:
                self._positive_cnt += 1
                results.append(1)
            else:
                results.append(0)
            self._total_cnt += 1
        predictions = np.array(predictions)
        results = np.array(results)

        plt.figure(figsize=(5,5))
        plt.scatter(X_test[:, 0][results[:]==0], X_test[:, 1][results[:]==0], marker='x', c='r')
        plt.scatter(X_test[:, 0][results[:]==1], X_test[:, 1][results[:]==1], marker='*', c='b')
        plt.legend(["Mispredicted", "Right"])
        plt.savefig(os.path.join(save_path, index + "_visual_test.png"), dpi=100)
        print(f"Test Accuracy: {self._positive_cnt/self._total_cnt}.")


    def polt_decision_boundary(self, model: BaseModel, scale: int=200, save_path: str="figures/SingleModel", index: str="0"):
        assert scale >= 0
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        grid_points = []
        for i in np.linspace(-6, 6, scale):
            for j in np.linspace(-6, 6, scale):
                grid_points.append(np.expand_dims(np.array([i, j]), axis=0))
        grid_points = np.concatenate(grid_points, axis=0)

        preds = np.round(self.predict(model, grid_points))
        
        plt.figure(figsize=(5,5))
        plt.scatter(grid_points[:, 0][preds[:, 0]==0], grid_points[:, 1][preds[:, 0]==0], c='k')
        plt.scatter(grid_points[:, 0][preds[:, 0]==1], grid_points[:, 1][preds[:, 0]==1], c='w')

        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(index + " decision boundary")
        plt.legend(["Label=0", "Label=1"])
        plt.savefig(os.path.join(save_path, index + "_decision_boundary.png"), dpi=200)










class MinMaxTrainer(object):
    def __init__(self, models: List, lr: float=1e-2, multiprocessing=False):
        self.workers = 9
        self.multiprocessing = multiprocessing
        self.models = models
        if self.multiprocessing:
            self.actual_workers = min(self.workers, os.cpu_count())
            self.trainers = [Trainer() for i in range(self.workers)]
            self.optimizers = [Optimizer(lr=lr) for i in range(self.workers)]
        else:
            self.actual_workers = 1
        self.trainer = Trainer()
        self.optimizer = Optimizer(lr=lr)


    def save_models(self, save_path: str="ckpts", index: str="0"):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        model_path = os.path.join(save_path, index+"_MinMax_models.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.models, f)
        print(f"MinMax models successfully saved in '{model_path}'")
    
    def load_models(self, model_path):
        if not os.path.isfile(model_path):
            raise ValueError(f"Model path '{model_path}' is NOT a file!")
        
        with open(model_path, "rb") as f:
            self.models = pickle.load(f)
        
        print(f"Successfully load models from '{model_path}'!")


    def train(self, data: TwoSpiralDataSet, strategy: str="RP",  epochs: int=1, early_stop: bool=True):
        assert strategy in ["RP", "PK"]
        if strategy == "RP":
            data.load_RP_subsets() 
        else:
            data.load_PK_subsets()


        if not self.multiprocessing:
            for i in tqdm(range(self.workers)):
                self.trainer.train(self.models[i], optimizer=self.optimizer, X_train=data.X_train_subsets[i], Y_train=data.Y_train_subsets[i], epochs=epochs, verbose=False, early_stop=early_stop)
                
        else:
            print(f"Using multiprocessing with {self.actual_workers} workers")
            results = []
            start_time = time.time()
            with Pool(processes=self.actual_workers) as pool:
                for i in range(self.workers):
                    results.append(pool.apply_async(self.trainers[i].train, (self.models[i], self.optimizers[i], data.X_train_subsets[i], data.Y_train_subsets[i], epochs, False, early_stop)))
                pool.close()
                pool.join()
                for i in range(self.workers):
                    self.models[i] = results[i].get()
                pool.terminate()
            print(f"Traing time with Multiprocessing: {time.time()-start_time}s")

            


    def predict(self, X: np.ndarray):
        # X.shape = (batchsize, 2)

        predictions = []
        min_predictions = []
        for i in range(self.workers):
            predictions.append(self.trainer.predict(self.models[i], X))
        predictions = np.concatenate(predictions, axis=1).transpose()
        # predictions.shape = (9, batchsize)

        min_predictions.append(np.min(predictions[:3, :], axis=0).reshape(-1, 1))
        min_predictions.append(np.min(predictions[3:6, :], axis=0).reshape(-1, 1))
        min_predictions.append(np.min(predictions[6:, :], axis=0).reshape(-1, 1))
        max_predictions = np.max(np.concatenate(min_predictions, axis=1), axis=1).reshape(-1, 1)
        # max_predictions.shape = (batchsize, 1)

        return np.round(max_predictions)

    def test(self, data: TwoSpiralDataSet):
        data.load_test_data()
        max_predictions = self.predict(data.X_test)

        correct_cnt = 0
        total_cnt = len(max_predictions)
        for i in range(total_cnt):
            prediction = np.round(max_predictions[i])
            label = data.Y_test[i]
            if prediction == label:
                correct_cnt += 1
        print(f"Test Accuracy = {correct_cnt/total_cnt}")


    def plot_decision_boundary(self, scale: int=200, save_path: str="figures/MinMax", index: str="0"):

        assert scale >= 0
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        grid_points = []
        for i in np.linspace(-6, 6, scale):
            for j in np.linspace(-6, 6, scale):
                grid_points.append(np.expand_dims(np.array([i, j]), axis=0))
        grid_points = np.concatenate(grid_points, axis=0)

        preds = self.predict(grid_points)

        plt.figure(figsize=(5,5))
        plt.scatter(grid_points[:, 0][preds[:, 0]==0], grid_points[:, 1][preds[:, 0]==0], c='k')
        plt.scatter(grid_points[:, 0][preds[:, 0]==1], grid_points[:, 1][preds[:, 0]==1], c='w')
        plt.xlim([-6,6])
        plt.ylim([-6,6])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(index + " MinMax decision boundary")
        plt.savefig(os.path.join(save_path, index + "_MinMax_decision_boundary.png"), dpi=200)


    def plot_submodel_decision_boundary(self, scale: int=100, save_path: str="figures/MinMax", index: str="0"):
        grid_points = []
        for i in np.linspace(-6, 6, scale):
            for j in np.linspace(-6, 6, scale):
                grid_points.append(np.expand_dims(np.array([i, j]), axis=0))
        grid_points = np.concatenate(grid_points, axis=0)

        for i in range(9):   
            self.trainer.polt_decision_boundary(self.models[i], scale=100, save_path = save_path, index=index + "_submodel_" + str(i))

    def test_submodels(self, data: TwoSpiralDataSet, save_path: str="figures/MinMax", index: str="0"):
        for i in range(self.workers):
            # self.trainer.test(self.models[i], data.X_test, data.Y_test)
            self.trainer.visual_test(self.models[i], data.X_train_subsets[i], data.Y_train_subsets[i], save_path=save_path, index=index + "_submodel_" + str(i))







            

















