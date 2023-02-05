import os
from datetime import datetime

splitter = '-' * 50 + '\n'

class Logger(object):
    def __init__(self, expdir):
        self.expdir = expdir
        self.loggingFile = os.path.join(expdir, "Train.log")
        self.evalFile = os.path.join(expdir, "Eval.log")

    def write(self, step, loss, valLoss, records, lr, grad_norm):
        with open(self.loggingFile, 'a') as f:
            f.write(f"Step {step}\n")
            f.write(f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            f.write(splitter)

            f.write(f"Loss: {loss} \n")
            f.write(f"Average Validation Loss: {valLoss} \n")
            
            f.write(f"Values: \n")
            for key, values in records.items():
                f.write(f"   {key}: {values} \n")

            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Gradient Norm: {grad_norm}\n\n")

        print(f"Step {step} \t Validation Loss: {valLoss}")

    def asrWrite(self, split, loss, uer, wer, step):
        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = (f"Step: {step}\n"
                f"{t}\n"
                f"{splitter}"
                f'asr/{split}-loss: {loss}\n'
                f'asr/{split}-uer: {uer} \n'
                f'asr/{split}-wer: {wer} \n\n'
                )
        
        if split == 'train':
            with open(self.loggingFile, 'a') as f:
                f.write(info)
        else:
            with open(self.evalFile, 'a') as f:
                f.write(info)

        print(info)
    
    def prWrite(self, split, acc, step):
        prefix = f'libri_phone/{split}-acc'
        t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        info = (f"Step: {step}\n"
                f"{t}\n"
                f"{splitter}"
                f'{prefix}: {acc}\n\n'
                )

        if split == 'train':
            with open(self.loggingFile, 'a') as f:
                f.write(info)
        else:
            with open(self.evalFile, 'a') as f:
                f.write(info)

        print(info)

        
