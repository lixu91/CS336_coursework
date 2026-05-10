from .llm_modules import Transformer,Softmax
from .train_module import crs_enty_ls,AdamW, cos_annealing,grad_clip,data_loader, save_checkpoint,load_checkpoint
from .adapters import get_tokenizer
import pickle
import numpy as np
import argparse
import torch
def main():
    train_dataset = np.memmap('./saved_tokenizer/TS_train_token_unit16.bin',dtype= np.uint16, mode='r')
    valid_dataset = np.memmap('./saved_tokenizer/TS_valid_token_unit16.bin',dtype= np.uint16, mode='r')
    parser = argparse.ArgumentParser(description='train-script')
    parser.add_argument("--batch_size", type= int, default= 32)
    parser.add_argument("--d_model", type= int, default= 512)
    parser.add_argument("--num_layers", type= int, required= True)
    parser.add_argument("--num_heads", type=int ,required= True)
    parser.add_argument("--d_ff", type= int, default= None)
    parser.add_argument("--rope_theta", type= float, required= True)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2",type= float,default=0.98)
    parser.add_argument("--lr",type=float, default=1e-5)
    parser.add_argument('--eps',type= float,default=1e-8)
    parser.add_argument('--weight_decay',type=float,default=0.0001)
    parser.add_argument('--context_length', type= int, required= True)
    parser.add_argument('--epoch', type=int ,required= True)
    parser.add_argument('--generate_mode', type=bool, required=True)
    args = parser.parse_args()

    vocab_size = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_l2_norm = 1.0
    betas = (args.beta1,args.beta2)
    #preparing dataset
    transformer_model = Transformer(vocab_size= vocab_size,d_model=args.d_model,num_heads=args.num_heads, 
                        context_length= args.context_length,num_layers= args.num_layers,d_ff = args.d_ff, 
                        rope_theta= args.rope_theta, device = device)
    optimizer = AdamW(transformer_model.parameters(),lr = args.lr, 
                            betas= betas, eps= args.eps, weight_decay= args.weight_decay)
    epochs = args.epoch
    loss_history = []
    valoss_history =[]
    valoss_history.append(float("inf"))
    train_loss = float('inf')
    valid_loss = float('inf')
    if not args.generate_mode :
        for step in range(epochs):
            if step >1 and step % 100 == 0:
                transformer_model.eval()
                with torch.no_grad():
                    valid_inputs, valid_target = data_loader(dataset=valid_dataset,batch_size=args.batch_size, 
                                context_length=args.context_length, device = device)
                    valid_logtis = transformer_model(valid_inputs)
                    valid_loss = crs_enty_ls(pre_logits= valid_logtis, targets= valid_target)
                    valid_loss_data = float(valid_loss.item())
                    if valid_loss_data < min(valoss_history):
                        save_checkpoint(model=transformer_model,optimizer=optimizer,iteration= step,out= "./checkpoints/best.pt")
                    valoss_history.append(valid_loss_data)
                transformer_model.train()

            train_inputs, train_target = data_loader(dataset=train_dataset,batch_size=args.batch_size, 
                                context_length=args.context_length, device = device)
            train_logits = transformer_model(train_inputs)
            train_loss = crs_enty_ls(pre_logits=train_logits, targets=train_target)
            train_loss.backward()
            grad_clip(transformer_model.parameters(),max_l2_norm= max_l2_norm)
            lr = cos_annealing(step, lr_max= 0.0003, lr_min= args.lr, T_w = 500,T_c=round(args.epoch * 0.8))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(float(train_loss.item()))
            print(f"Epoch {step +1:03d} | train_loss = {train_loss:.6f} | valid_loss = {valid_loss:.6f}")
    else:generate_text(input_text="everyone races", model=transformer_model, para_path="./checkpoints/best.pt", optimizer= optimizer, max_gen_len=100)

def generate_text(input_text : str, model, para_path, optimizer, max_gen_len:int,t:float = 0.98, p :float= 0.8):
    with open("./saved_tokenizer/vocab.pkl",'br') as f:
        vocab = pickle.load(f)

    with open("./saved_tokenizer/special_tokens.pkl",'br') as f:
        special_token = pickle.load(f)

    with open("./saved_tokenizer/merges.pkl",'br') as f:
        merge = pickle.load(f)
    special_id = 256
    TinyStorie_tokenizer = get_tokenizer(vocab,merge,special_token)
    load_checkpoint(para_path, model= model, optimizer= optimizer)
    model.eval()
    with torch.no_grad():
        input_seq = TinyStorie_tokenizer.encode(input_text)
        input_seq_tensor = torch.tensor(input_seq)
        gen_len = 0
        while gen_len <= max_gen_len:
            logits = model(input_seq_tensor) / t
            logits = Softmax(logits, dim=-1)
            last_logit = logits[-1]
            sum = 0.0
            counts = 0
            sorted_logits, orig_index = torch.sort(last_logit,descending=True)
            for i in range(len(last_logit)):
                if sum > p:
                    break
                sum += sorted_logits[i].item()
                counts += 1
            sel_logit = sorted_logits[:counts]
            top_tokens = sel_logit / torch.sum(sel_logit)
            Sample_index = torch.multinomial(top_tokens, num_samples=1)
            Sample_index = Sample_index.item()
            out_put_token = orig_index[Sample_index]
            if out_put_token.item() == 256:
                break
            input_seq.append(out_put_token.item())
            gen_len +=1
        print(TinyStorie_tokenizer.decode(input_seq))










if __name__ == "__main__":
    main()