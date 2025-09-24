
import torch
from utils import SEEDLoaderH, SEEDLoader,get_bci4_cross, get_bci4_within
import numpy as np
import torch.nn as nn
import os
import time
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR
from model import Baseline_seed, Baseline_bci4_v2
from loss import  FMCA, FMCA_PosNeg
from tqdm import tqdm
from collections import Counter
import pickle
import wandb
from datasets.seed import SEED_ss, SEED_ssH, prepare_seed

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# evaluation design
def task(X_train, X_test, y_train, y_test, n_classes):       
    cls = LR(solver='lbfgs', multi_class='multinomial', max_iter=500)
    len_y = len(y_train)
    cls.fit(X_train, y_train) #   [:len_y], y_train)
    len_y_2 = len(y_test)
    pred = cls.predict(X_test) # [:len_y_2])
    res = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    
    return res, cm

def Pretext(q_encoder, k_encoder, optimizer, Epoch, criterion, pretext_loader, train_loader, test_loader, save_dir):
    q_encoder.train(); 
    global queue
    global queue_ptr
    global n_queue
    step = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.steplr_point, gamma=0.2)
    acc = evaluate(q_encoder, train_loader, test_loader, step)
    print("Untrained model ACC, ", acc)
    all_loss, acc_score = [], []
    nan_count = 0
    best_acc = 0
    
    for epoch in range(Epoch):
        epoch_loss = 0
        epoch_steps = 0
        
        for index, (aug1, aug2) in enumerate(pretext_loader): 
            aug1, aug2 = aug1.to(device), aug2.to(device)
            emb_aug1, emb_aug2 = q_encoder(aug2, train=True)
            if args.model == 'FMCA':
                loss = criterion(emb_aug1, emb_aug2, step)
            else:
                loss = criterion(emb_aug1, emb_aug2)

            epoch_loss += loss.item()
            epoch_steps += 1
            all_loss.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # only update encoder_q
            
            N = 1000
            if (step + 1) % N == 0: #  and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                acc = evaluate(q_encoder, train_loader, test_loader, step)
                acc_score.append([sum(all_loss[-N:]) / len(all_loss[-N:]), acc])
                if acc > best_acc:
                    best_acc = acc
                
            step += 1

        acc = evaluate(q_encoder, train_loader, test_loader, step)
        acc_score.append([sum(all_loss[-N:]) / len(all_loss[-N:]), acc])
        
        if acc > best_acc:
            best_acc = acc

        avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
        print(f'Epoch: {epoch}, Average Loss: {avg_epoch_loss:.6f}')

        scheduler.step()
    
    print()
    print("="*50)
    print(f"Training completed! Best accuracy: {best_acc:.4f}")
    print("="*50)

def evaluate(q_encoder, train_loader, test_loader, step):

    q_encoder.eval()

    emb_val, gt_val = [], []
    count = 0
    with torch.no_grad():
        for (X_val, y_val) in train_loader:
            X_val = X_val.to(device)
            emb_val.extend(q_encoder(X_val, step=step)[0].cpu().tolist())
            gt_val.extend(y_val.numpy().flatten())
            count += 1

    emb_val, gt_val = np.array(emb_val), np.array(gt_val)


    emb_test, gt_test = [], []
    with torch.no_grad():
        for (X_test, y_test) in test_loader:
            X_test = X_test.to(device)
            emb_test.extend(q_encoder(X_test,step=step)[0].cpu().tolist())
            gt_test.extend(y_test.numpy().flatten())
    emb_test, gt_test= np.array(emb_test), np.array(gt_test)
           
    emb_val = np.nan_to_num(emb_val, nan=0.0)  
    emb_test = np.nan_to_num(emb_test, nan=0.0)

    res, cm = task(emb_val, emb_test, gt_val, gt_test, 5)
    q_encoder.train()
    return res

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help="number of epochs")
    parser.add_argument('--lr', type=float, default=3e-5, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=64, help="hidden units (for SHHS, 256, for Sleep, 128)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
    parser.add_argument('--model', type=str, default='FMCA', help="which model")
    parser.add_argument('--T', type=float, default=0.3,  help="T")
    parser.add_argument('--sigma', type=float, default=2.0,  help="sigma")
    parser.add_argument('--delta', type=float, default=0.2,  help="delta")
    parser.add_argument('--dataset', type=str, default='SEED', help="dataset")
    parser.add_argument('--testsub', type=str, default='1', help="dataset")
    parser.add_argument('--steplr_point', type=int, default=20, help="seed 20, bci 200")
    parser.add_argument('--temperature',type=float,default=0.07)
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--beta',type=float,default=0.5)

    args = parser.parse_args()


    dataset = args.dataset
    model = args.model
    n_dim = args.n_dim
    batch_size = args.batch_size
    testsub = args.testsub
    lr = float(args.lr)
    weight_decay = float(args.weight_decay)
    alpha = args.alpha
    beta = args.beta


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    # set random seed
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    global queue
    global queue_ptr
    global n_queue

    print("Current training dataset ", args.dataset)



    if args.dataset == 'SEED':

        pretext_loader = torch.utils.data.DataLoader(SEEDLoaderH("seed", test_subject=args.testsub, train=True, SS=True),
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SEEDLoader("seed", test_subject=args.testsub, train=True, SS=False),
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SEEDLoader("seed", test_subject=args.testsub, train=False, SS=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define the model
        q_encoder = Baseline_seed(args.n_dim)
        q_encoder.to(device)

        k_encoder = None 



    elif args.dataset == 'SEED_newread':
        data_path = "/projects/EEG-foundation-model/SEED"
        split_mode="strict"
        test_size = 0.15
        window_sec = 2
        print("Current test subject is : ", args.testsub)
        # test subject is -1 means split without subjects
        train_x, train_y, test_x, test_y = prepare_seed(data_path, args.testsub, split_mode=split_mode, 
                                                        test_size=test_size, sfreq=200, window_sec=window_sec, read_mode="tail")


        pretext_loader = torch.utils.data.DataLoader(SEED_ssH(train_x, train_y,ss=True),
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(SEED_ss(train_x, train_y, ss=False),
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        test_loader = torch.utils.data.DataLoader(SEED_ss(test_x, test_y, ss=False),
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        # define the model
        q_encoder = Baseline_seed(args.n_dim)
        # q_encoder = Baseline_seed_v3(args.n_dim)
        q_encoder.to(device)

        k_encoder = None 




    elif args.dataset == 'BCI-IV-2A':

        data_path = "./BCIC_2a_0_38HZ/"
        # get_bci4_cross, get_bci4_within
        if int(args.testsub) == -1:
            pretext_dataset, train_dataset, test_dataset = get_bci4_within(data_path, hireac=True, twoA=True)
        else:
            assert  0 < int(args.testsub) < 10 # "BCI-IV should samller than 10"
            pretext_dataset, train_dataset, test_dataset = get_bci4_cross(int(args.testsub), data_path, hireac=True, twoA=True)

        print('pretext (all patient): ', len(pretext_dataset))
        print('train (all patient): ', len(train_dataset))
        print('test (all) patient): ', len(test_dataset))
        print("SSL train data shape example: ", pretext_dataset[1][1].shape)

        pretext_loader = torch.utils.data.DataLoader(pretext_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                        batch_size=args.batch_size, shuffle=True, num_workers=20)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                        batch_size=args.batch_size, shuffle=False, num_workers=20)

        q_encoder = Baseline_bci4_v2(args.n_dim)
        q_encoder.to(device)

        k_encoder = None 

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=weight_decay)

    if args.model == 'HFMCA':
        criterion = FMCA(device, use_trace=False, hirea=True).to(device)
    elif args.model == 'HFMCA+':
        criterion = FMCA_PosNeg(device,temperature=args.temperature, alpha=args.alpha, beta=args.beta).to(device)


        # criterion = HFMCA(device, use_trace=True, hirea=True).to(device)
    model_name = 'HFMCA'
    saved_models_dir = f"{args.dataset}_{args.model}_dim{args.n_dim}_bs{args.batch_size}_sub{args.testsub}_aug_4"
    os.makedirs(saved_models_dir, exist_ok=True)
    # optimize
    starttt = time.time()
    Pretext(q_encoder, k_encoder, optimizer, args.epochs, criterion, pretext_loader, train_loader, test_loader,saved_models_dir)
    print("Time cost: ", (time.time()-starttt)/360)
