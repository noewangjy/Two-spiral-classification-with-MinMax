import argparse
import os
from data_utils import TwoSpiralDataSet
from modeling import OneLayerMLQPwithLinearOut, TwoLayerMLQP, ThreeLayerMLQP, TwoLayerMLQPwithLinearOut
from optimizer import Optimizer
from trainer import Trainer



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OneLayerMLQPwithLinearOut", 
                        help="Valid model names:'OneLayerMLQPwithLinearOut', 'TwoLayerMLQP', 'TwoLayerMLQPwithLinearOut'. 'ThreeLayerMLQP'.")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--activation", type=str, default="sigmoid")   
    parser.add_argument("--input_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=40)
    parser.add_argument("--hidden_size2", type=int, default=0)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--save_path", type=str, default="./ckpts", help="The directory to save model") # save model to dir: save_path
    parser.add_argument("--model_path", type=str, default=None, help="The file to load model from.")   # load model from file: model_path
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--visualize_db", action="store_true")
    parser.add_argument("--early_stop", action="store_true") 
    args = parser.parse_args()

    
    if args.model_path is None:
        args.model_path = os.path.join(args.save_path, args.model_name + "_" +args.activation + "_" + str(args.hidden_size) + "_model.pkl")


    data = TwoSpiralDataSet(data_path=args.data_path)
    data.load_data()

    if args.model_name == "OneLayerMLQPwithLinearOut":
        model = OneLayerMLQPwithLinearOut(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, activation=args.activation)
    elif args.model_name == "TwoLayerMLQP":
        model = TwoLayerMLQP(input_size=args.input_size, hidden_size=args.hidden_size, output_size=args.output_size, activation=args.activation)
    elif args.model_name == "ThreeLayerMLQP":
        model = ThreeLayerMLQP(input_size=args.input_size, h1=args.hidden_size, h2=args.hidden_size2, output_size=args.output_size, activation=args.activation)
    elif args.model_name == "TwoLayerMLQPwithLinearOut":
        model = TwoLayerMLQPwithLinearOut(input_size=args.input_size, h1=args.hidden_size, h2=args.hidden_size2, output_size=args.output_size, activation=args.activation)
    else:
        raise ValueError("Invalid model_name!")
    


    optimizer = Optimizer(lr=args.learning_rate)


    trainer = Trainer()
    if args.do_train:
        trainer.train(model, optimizer=optimizer, X_train=data.X_train, Y_train=data.Y_train, epochs=args.epochs, early_stop=args.early_stop)
        trainer.save_model(model, args.save_path, index=args.model_name+"_"+args.activation+"_" + str(args.hidden_size))
    if args.do_test:
        model = trainer.load_model(model_path=args.model_path)
        trainer.visual_test(model, X_test=data.X_test, Y_test=data.Y_test, index=args.model_name+"_"+args.activation)
    if args.visualize_db:
        model = trainer.load_model(model_path=args.model_path)
        trainer.polt_decision_boundary(model, index=args.model_name+"_"+args.activation)




                    
















        

        





            








    

