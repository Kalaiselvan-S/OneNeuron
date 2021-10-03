from utils.all_utils import save_model, save_plot
from utils.models import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(lineno)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir,"running_logs.log"), level=logging.INFO, format=logging_str)





def main(data, modeLName, plotName, eta, epochs):

    df = pd.DataFrame(AND)

    logging.info(f"This is the actual df{df}")


    X,y = prepare_data(df)

    ETA = 0.3 # 0 and 1
    EPOCHS = 10

    model = Perceptron(eta=ETA, epochs=EPOCHS)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename='and.model')
    save_plot(df, 'and.png', model)

if __name__ == '__main__':
    AND = { 
        "x1": [0,0,1,1],
        "x2": [0,1,0,1],
        "y": [0,0,0,1],
    }
    ETA = 0.3
    EPOCHS = 10
    try:
        logging.info(">>> start the module >>>")
        main(data=AND, modeLName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
    except Exception as e:
        logging.exception(e)
        raise e
