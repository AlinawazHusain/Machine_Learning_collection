import config
import torch.nn as nn
import dataset
import utils
import generator
import descriminator
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm 



def train_fn(gen , desc , gen_opt , desc_opt ,loader ,  bce , l1 ,g_scal , d_scal, epoch ):
    p_bar = tqdm(loader , desc = f'EPOCH {epoch+1}/{config.EPOCH-1}')

    for batch_idx , (x , y ) in enumerate(p_bar):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)


        with torch.cuda.amp.autocast():
             fake = gen(x)
             desc_real = desc(x ,y)
             desc_fake = desc(x , fake.detach())

             desc_real_loss = bce(desc_real , torch.ones_like(desc_real))
             desc_fake_loss = bce(desc_fake , torch.zeros_like(desc_fake))

             desc_loss = (desc_real_loss + desc_fake_loss)/2

             desc.zero_grad()
             d_scal.scale(desc_loss).backward()
             d_scal.step(desc_opt)
             d_scal.update()

        
        with torch.cuda.amp.autocast():

            output = desc(x , fake)
            loss1 = bce(output , torch.ones_like(output))
            loss2 = l1(fake , y) * config.L1_LAMBDA

            gen_loss = loss1 + loss2

            gen_opt.zero_grad()
            g_scal.scale(gen_loss).backward()
            g_scal.step(gen_opt)
            g_scal.update()


        p_bar.set_postfix({"gen_loss" : gen_loss.item() , "desc_loss" : desc_loss.item()})



def main():
    desc = descriminator.Descriminator().to(config.DEVICE)
    gen = generator.Generator().to(config.DEVICE)

    gen_opt = optim.Adam(gen.parameters() , lr = config.LEARNING_RATE , betas = (0.5 , 0.999))
    desc_opt = optim.Adam(desc.parameters() , lr = config.LEARNING_RATE , betas = (0.5 , 0.999))

    bce_loss = nn.BCELoss().to(config.DEVICE)
    l1_loss = nn.L1Loss().to(config.DEVICE)

    if config.LOAD_MODEL:
        utils.load_pretrained(gen , desc , config.LEARNING_RATE , gen_opt , desc_opt)


    training_dataset = dataset.Custdata(config.TRAIN_DIR)
    train_loader = DataLoader(training_dataset , config.BATCH_SIZE , shuffle = True , num_workers = config.NUM_WORKERS)

    val_dataset = dataset.Custdata(config.VAL_DIR)
    val_loader = DataLoader(val_dataset , config.BATCH_SIZE , shuffle = False)

    g_scaler = torch.cuda.amp.grad_scaler()
    d_scaler = torch.cuda.amp.grad_scaler()

    gen.train()
    desc.train()
    for epoch in len(config.EPOCH):
        train_fn(gen , desc ,gen_opt , desc_opt ,train_loader , bce_loss , l1_loss , g_scaler , d_scaler , epoch)


        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_model(gen , desc , gen_opt , desc_opt , epoch)
    
    utils.save_examples(gen , val_loader , epoch , 'Saved_Examples')


if __name__ == '__main__':
    main()
