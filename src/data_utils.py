import os
import matplotlib.pyplot as plt
import numpy as np


class TwoSpiralDataSet(object):
    def __init__(self, data_path="data") -> None:
        self.train_path = os.path.join(data_path, "two_spiral_train_data-1.txt")
        self.test_path = os.path.join(data_path, "two_spiral_test_data-1.txt")
        
        if not os.path.exists(self.train_path):
            raise ValueError(f"Train data NOT found in {self.train_path}!")
        if not os.path.exists(self.test_path):
            raise ValueError(f"Test data NOT found in {self.test_path}!")

        self._X_train = None
        self._Y_train = None
        self._X_test = None
        self._Y_test = None
        self._X_train_subsets = []
        self._Y_train_subsets = []
        self._subset_strategy = None
        self.figure_path = "figures/data"

    @property
    def X_train(self):
        return self._X_train

    @property
    def Y_train(self):
        return self._Y_train
    
    @property
    def X_test(self):
        return self._X_test

    @property
    def Y_test(self):
        return self._Y_test

    @property
    def X_train_subsets(self):
        return self._X_train_subsets
    
    @property
    def Y_train_subsets(self):
        return self._Y_train_subsets


    def load_train_data(self):
        train_data = np.loadtxt(self.train_path)
        self._X_train = train_data[:, :2]
        self._Y_train = train_data[:, 2]
        return self._X_train, self._Y_train

    def load_test_data(self):
        test_data = np.loadtxt(self.test_path)
        self._X_test = test_data[:, :2]
        self._Y_test = test_data[:, 2]   
        return self._X_test, self._Y_test    

    def load_data(self):
        self.load_train_data()
        self.load_test_data()


    def load_RP_subsets(self, visualize=False):
        save_path = self.figure_path
        if len(self._X_train_subsets) != 0 and len(self.Y_train_subsets) != 0 and self._subset_strategy == "RP":
            return self._X_train_subsets, self._Y_train_subsets
        else:
            self._X_train_subsets = []
            self._Y_train_subsets = []
            self._subset_strategy = "RP"
        
        partitions = 3


        self.load_train_data()
        self.load_test_data()

        # Filter postive samples
        pos_samples = self._X_train[self._Y_train==1]
        pos_indices = np.arange(pos_samples.shape[0])
        

        # Radomly divide positive samples into `partitions` folds
        pos_subsets = []
        unsampled_indices = np.arange(len(pos_indices))
        samples_per_subset = int(np.floor(pos_samples.shape[0]/partitions))
        for i in range(partitions-1):
            idx = np.random.choice(unsampled_indices, samples_per_subset, replace=False)
            pos_subsets.append(pos_samples[pos_indices[idx], :])
            pos_indices = np.delete(pos_indices, idx)
            unsampled_indices = np.arange(len(pos_indices))
        pos_subsets.append(pos_samples[pos_indices, :])

        # Filter negative samples
        neg_samples = self._X_train[self._Y_train==0]
        neg_indices = np.arange(neg_samples.shape[0])

        neg_subsets = []
        unsampled_indices = np.arange(len(neg_indices))
        samples_per_subset = int(np.floor(neg_samples.shape[0]/partitions))
        for i in range(partitions-1):
            idx = np.random.choice(unsampled_indices, samples_per_subset, replace=False)
            neg_subsets.append(neg_samples[neg_indices[idx], :])
            neg_indices = np.delete(neg_indices, idx)
            unsampled_indices = np.arange(len(neg_indices))
        neg_subsets.append(neg_samples[neg_indices, :])

        # Cross composition of positive and negative samples to get partition^2 subsets
        for i in range(partitions):
            for j in range(partitions):
                X_temp = np.concatenate([pos_subsets[i], neg_subsets[j]], axis=0)
                Y_temp = np.array([1] * len(pos_subsets[i]) + [0] * len(neg_subsets[j]))
                idx = np.arange(len(Y_temp))
                np.random.shuffle(idx)
                self._X_train_subsets.append(X_temp[idx, :])
                self._Y_train_subsets.append(Y_temp[idx])
        
        assert len(self._X_train_subsets) == partitions**2
        assert len(self._Y_train_subsets) == partitions**2

        if visualize:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            fig, axes = plt.subplots(1, 2, figsize=(11,5))
            for i in range(partitions):
                axes[0].scatter(pos_subsets[i][:, 0], pos_subsets[i][:, 1])
            axes[0].legend([f"Subset {i}" for i in range(partitions)])
            axes[1].scatter(self._X_train[:, 0][self._Y_train==1], self._X_train[:, 1][self._Y_train==1])
            plt.savefig(os.path.join(save_path, "RP_positive.png"), dpi=100)


            fig, axes = plt.subplots(1, 2, figsize=(11,5))
            for i in range(partitions):
                axes[0].scatter(neg_subsets[i][:, 0], neg_subsets[i][:, 1])
            axes[0].legend([f"Subset {i}" for i in range(partitions)])
            axes[1].scatter(self._X_train[:, 0][self._Y_train==0], self._X_train[:, 1][self._Y_train==0])
            plt.savefig(os.path.join(save_path, "RP_negative.png"), dpi=100)

        return self._X_train_subsets, self._Y_train_subsets



    def load_PK_subsets(self, visualize=False):
        save_path = self.figure_path
        if len(self._X_train_subsets) != 0 and len(self.Y_train_subsets) != 0 and self._subset_strategy == "PK":
            return self._X_train_subsets, self._Y_train_subsets
        else:
            self._X_train_subsets = []
            self._Y_train_subsets = []
            self._subset_strategy = "PK"

        partitions = 3
        

        self.load_train_data()
        self.load_test_data()

        # Filter postive samples
        pos_samples = self._X_train[self._Y_train==1]
        # Divide positive samples into `partitions` folds with prior knowledge
        pos_subsets = [[], [], []]
        for i in range(pos_samples.shape[0]):
            X = pos_samples[i, :]
            if X[0] < -2:
                pos_subsets[0].append(np.expand_dims(X, axis=0))
            elif X[0] > 2:
                pos_subsets[2].append(np.expand_dims(X, axis=0))
            else:
                pos_subsets[1].append(np.expand_dims(X, axis=0))


        # Filter negative samples
        neg_samples = self._X_train[self._Y_train==0]
        # Divide negative samples into `partitions` folds with prior knowledge
        neg_subsets = [[], [], []]
        for i in range(neg_samples.shape[0]):
            X = neg_samples[i, :]

            if X[0] < -2:
                neg_subsets[0].append(np.expand_dims(X, axis=0))
            elif X[0] > 2:
                neg_subsets[2].append(np.expand_dims(X, axis=0))
            else:
                neg_subsets[1].append(np.expand_dims(X, axis=0))


        for i in range(partitions):
            pos_subsets[i] = np.concatenate(pos_subsets[i], axis=0)
            neg_subsets[i] = np.concatenate(neg_subsets[i], axis=0)

        # Cross composition of positive and negative samples to get partition^2 subsets
        for i in range(partitions):
            for j in range(partitions):
                X_temp = np.concatenate([pos_subsets[i], neg_subsets[j]], axis=0)
                Y_temp = np.array([1] * len(pos_subsets[i]) + [0] * len(neg_subsets[j]))
                idx = np.arange(len(Y_temp))
                np.random.shuffle(idx)
                self._X_train_subsets.append(X_temp[idx, :])
                self._Y_train_subsets.append(Y_temp[idx])
        
        assert len(self._X_train_subsets) == partitions**2
        assert len(self._Y_train_subsets) == partitions**2

        if visualize:
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            fig, axes = plt.subplots(1, 2, figsize=(11,5))
            for i in range(partitions):
                axes[0].scatter(pos_subsets[i][:, 0], pos_subsets[i][:, 1])
            axes[0].legend([f"Subset {i}" for i in range(partitions)])
            axes[1].scatter(self._X_train[:, 0][self._Y_train==1], self._X_train[:, 1][self._Y_train==1])
            plt.savefig(os.path.join(save_path, "PK_positive.png"), dpi=100)


            fig, axes = plt.subplots(1, 2, figsize=(11,5))
            for i in range(partitions):
                axes[0].scatter(neg_subsets[i][:, 0], neg_subsets[i][:, 1])
            axes[0].legend([f"Subset {i}" for i in range(partitions)])
            axes[1].scatter(self._X_train[:, 0][self._Y_train==0], self._X_train[:, 1][self._Y_train==0])
            plt.savefig(os.path.join(save_path, "PK_negative.png"), dpi=100)

        return self._X_train_subsets, self._Y_train_subsets


    def visualize_subsets(self, X_subsets=None, Y_subsets=None):
        save_path = self.figure_path
        if X_subsets is None and Y_subsets is None:
            X_subsets = self._X_train_subsets
            Y_subsets = self._Y_train_subsets
        
        if self._subset_strategy is None:
            raise NotImplementedError("Subset NOT initialized! Please use '.load_RP_subsets()' or '.load_PK_subsets()' for initialization.")
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        partitions = int(np.sqrt(len(Y_subsets)))

        fig, axes = plt.subplots(partitions, partitions, figsize=(10,10))
        cnt = 0
        for i in range(partitions):
            for j in range(partitions):
                ax = axes[i, j]
                X = X_subsets[cnt]
                Y = Y_subsets[cnt]
                ax.scatter(X[:, 0][Y==0], X[:, 1][Y==0])
                ax.scatter(X[:, 0][Y==1], X[:, 1][Y==1])
                ax.set_xlim([-6, 6])
                ax.set_ylim([-6, 6])
                cnt += 1
        plt.savefig(os.path.join(save_path, self._subset_strategy + "_subsets.png"), dpi=100)



















        
            










