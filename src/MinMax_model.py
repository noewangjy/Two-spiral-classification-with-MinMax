import argparse
import os
from data_utils import TwoSpiralDataSet
from modeling import OneLayerMLQPwithLinearOut
from optimizer import Optimizer
from trainer import MinMaxTrainer





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_split_strategy", '-d', type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--activation", type=str, default="sigmoid")
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--hard_task", type=str, default="0,4,8", help="The index/indices of the hard sub-problem(s).")
    parser.add_argument("--h_easy", type=int, default=5, help="Hidden size of submodels with easy tasks")
    parser.add_argument("--h_hard", type=int, default=25, help="Hidden size of submodels with hard tasks")
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--save_path", type=str, default="./ckpts", help="The directory to save model") # save model to dir: save_path
    parser.add_argument("--model_path", type=str, default=None, help="The file to load model from.")   # load model from file: model_path
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--visualize_MinMax_db", action="store_true")
    parser.add_argument("--visualize_submodel_db", action="store_true") 
    parser.add_argument("--visualize_data_subsets", action="store_true")
    args = parser.parse_args()



    data = TwoSpiralDataSet(data_path=args.data_path)
    
    if args.data_split_strategy == "RP":
        data.load_RP_subsets(visualize=True)
        models = [OneLayerMLQPwithLinearOut(input_size=args.input_size, hidden_size=args.h_hard, output_size=args.output_size, activation=args.activation) for i in range(9)]
        
    elif args.data_split_strategy == "PK":
        data.load_PK_subsets(visualize=True)
        models = [OneLayerMLQPwithLinearOut(input_size=args.input_size, hidden_size=args.h_easy, output_size=args.output_size, activation=args.activation) for i in range(9)]
        for i in args.hard_task.split(","):
            models[int(i)] = OneLayerMLQPwithLinearOut(input_size=args.input_size, hidden_size=args.h_hard, output_size=args.output_size, activation=args.activation)


    if args.visualize_data_subsets:
        data.visualize_subsets()

    if args.model_path is None:
        args.model_path = os.path.join(args.save_path, args.data_split_strategy + "_MinMax_models.pkl")









    optimizer = Optimizer(lr=args.learning_rate)

    MinMaxtrainer = MinMaxTrainer(models=models, lr=args.learning_rate, multiprocessing=args.multiprocessing)

    if args.do_train:
        MinMaxtrainer.train(data=data, strategy=args.data_split_strategy, epochs=args.epochs)
        MinMaxtrainer.save_models(save_path=args.save_path, index=args.data_split_strategy)

    if args.do_test:
        MinMaxtrainer.load_models(model_path=args.model_path)
        MinMaxtrainer.test(data)

    if args.visualize_MinMax_db:
        MinMaxtrainer.load_models(model_path=args.model_path)
        MinMaxtrainer.plot_decision_boundary(index=args.data_split_strategy)

    if args.visualize_submodel_db:
        MinMaxtrainer.load_models(model_path=args.model_path)
        MinMaxtrainer.test_submodels(data, index=args.data_split_strategy)
        MinMaxtrainer.plot_submodel_decision_boundary(index=args.data_split_strategy)
