import torch
import config
from torchvision.utils import save_image

def save_examples(gen , val_loader , epoch , folder):
    x , y = next(iter(val_loader))
    x = x.to(config.DEVICE)
    y = y.to(config.DEVICE)

    gen.eval()
    with torch.no_grad():
        fake_y = gen(x)
        fake_y = fake_y*0.5 +0.5

        save_image(fake_y , folder+f"/y_gen_{epoch}.png")
        save_image(x*0.5 +0.5 , folder+f"/x_input_{epoch}.png")

        if epoch == 1:
            save_image(y , folder+f"/y_real_{epoch}.png")
        
    gen.train()



def save_model(gen , des , gen_opt , des_opt , epoch ):
    print("==>> Saving checkpoint")
    torch.save({
        "Generator":gen.state_dict(),
        "Descriminator": des.state_dict(),
        "Generator_optimizer" : gen_opt.state_dict(),
        "Descriminator_optimizer" : des_opt.state_dict(),
        "Generator_architecture" : gen,
        "Descriminator_architecture" : des
                } ,
                config.SAVE_PATH+f"/epoch{epoch}.pth.tar")
    



def load_pretrained(gen , des , lr , gen_opt , des_opt):
    checkpoint = torch.load(config.SAVE_PATH)
    gen.load_state_dict(checkpoint['Generator'])
    des.load_state_dict(checkpoint["Descriminator"])
    gen_opt.load_state_dict(checkpoint["Generator_optimizer"])
    des_opt.load_state_dict(checkpoint["Descriminator_optimizer"])

    for param_group in gen_opt.param_groups:
        param_group['lr'] = lr
    
    for param_group in des_opt.param_groups:
        param_group['lr'] = lr

