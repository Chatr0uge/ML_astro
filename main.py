from load_data import * 
from DDMP import *
from colorama import Fore, Style
from pprint import pprint
from scipy.stats import describe
from pandas import DataFrame
from rich.table import Table
from rich.console import Console

if __name__ == "__main__" : 
    
    print(Fore.RED + 'Loading DATA'.center(30, '-') +'\n' + Style.RESET_ALL)
    
    data = np.load("/media/andrea/Crucial X6/data.npy")
    infos = DataFrame(describe(data, axis = None))
    
    infos.insert(0, 'Stat', ['Nobs', 'Min-Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])   
    
    console = Console()
    table = Table('Dataset Infos')
    table.add_row(infos.to_string(float_format=lambda _: '{:.4f}'.format(_)))
    print('Oringal Data Infos before normalization and transformation')
    console.print(table)
    
    print('\n' + Fore.RED + 'Preprocessing DATA'.center(30, '-') +'\n' + Style.RESET_ALL)
    
    dataset = [transform((d-d.min())/abs(d-d.min()).max()) for d in data]
    dataset_p = [d.numpy() for d in dataset]
    
    infos = DataFrame(describe(np.array(dataset_p), axis = None))
    infos.insert(0, 'Stat', ['Nobs', 'Min-Max', 'Mean', 'Variance', 'Skewness', 'Kurtosis'])   
    table = Table('Dataset Processed Infos')
    table.add_row(infos.to_string(float_format=lambda _: '{:.4f}'.format(_)))
    console.print(table)
    
    loader = DataLoader(dataset, 64, shuffle=True)

    
    print(Fore.RED +'\n' + 'Training Phase ...'.center(30, '-') + Style.RESET_ALL +'\n')
        
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    n_steps, min_beta, max_beta = 1000, 10 ** -4, .02  # Originally used by the authors
    ddpm = MyDDPM(MyUNet(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta = max_beta, device = device)

    training_loop(ddpm, loader, 25, optim= torch.optim.Adam(ddpm.parameters(), lr  = .001), device = device)




