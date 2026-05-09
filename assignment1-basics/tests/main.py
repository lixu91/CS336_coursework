from .llm_modules import Transformer
from .train_module import crs_enty_ls,AdamW, cos_annealing,grad_clip,data_loader, save_checkpoint,load_checkpoint
import numpy as np
import argparse
import torch
def main():
    train_dataset = np.memmap('./saved_tokenizer/TS_train_token_unit16.pkl',dtype= np.uint16, mode='r')
    valid_dataset = np.memmap('./saved_tokenizer/TS_valid_token_unit16.pkl',dtype= np.uint16, mode='r')
    parser = argparse.ArgumentParser(description='train-script')
    parser.add_argument("--batch_size", type= int, default= 32)
    parser.add_argument("--d_model", type= int, default= 512)
    parser.add_argument("--num_layers", type= int, required= True)
    parser.add_argument("--num_heads", type=int ,required= True)
    parser.add_argument("--d_ff", type= int, default= None)
    parser.add_argument("--rope_theta", type= float, required= True)
    parser.add_argument("--betas", type=tuple(float,float), default=(0.9,0.99))
    parser.add_argument("--lr",type=float, default=1e-4)
    parser.add_argument('--eps',type= float,default=1e-8)
    parser.add_argument('--weight_decay',type=float,default=0.01)
    parser.add_argument('--context_length', type= int, required= True)
    parser.add_argument('--epoch', type=int ,required= True)
    args = parser.parse_args()

    vocab_size = 10000
    device = "cpu"
    max_l2_norm = 1,0
    warmup_iter = 20
    #preparing dataset
    transformer_model = Transformer(vocab_size= vocab_size,d_model=args.d_model,num_heads=args.num_heads, num_layers= args.num_layers,
                              d_ff = args.d_ff, rope_theta= args.rope_theta, device = device)
    epochs = args.epoch
    loss_history = []
    valoss_history =[]
    for epoch in range(epochs):
        train_loss = float('inf')
        valid_loss = float('inf')
        if epoch >1 and epoch % 10 == 0:
            transformer_model.eval()
            with torch.no_grad():
                valid_inputs, valid_target = data_loader(dataset=valid_dataset,batch_size=args.batch_size, 
                             context_length=args.context_length, device = device)
                valid_logtis = transformer_model(valid_inputs)
                valid_loss = crs_enty_ls(pre_logits= valid_logtis, targets= valid_target)
                valoss_history.append(valid_loss)
                optimizer = AdamW(transformer_model.parameters(),lr = lr, 
                                  betas= args.betas, eps= args.eps, weight_decay= args.weight_decay)
                if valid_loss < min(valoss_history):
                    save_checkpoint(model=transformer_model,optimizer=optimizer,iteration= epoch,out= "./checkpoints")

        train_inputs, train_target = data_loader(dataset=train_dataset,batch_size=args.batch_size, 
                             context_length=args.context_length, device = device)
        train_logits = transformer_model(train_inputs)
        train_loss = crs_enty_ls(pre_logits=train_logits, targets=train_target)
        train_loss.backward()
        grad_clip(transformer_model.parameters(),max_l2_norm= max_l2_norm)
        lr = cos_annealing(epoch, lr_max= 0.001, lr_min= args.lr, T_w = warmup_iter,T_c=60)
        optimizer = AdamW(transformer_model.parameters(),lr = lr, betas= args.betas, eps= args.eps, weight_decay= args.weight_decay)
        optimizer.step()
        optimizer.zero_grad()
        loss_history.append(train_loss)
        print(f"Epoch {epoch +1:03d} | train_loss = {train_loss:.6f} | valid_loss = {valid_loss:.6f}")
    

