import skvideo
# skvideo.setFFmpegPath("/usr/bin")
import skvideo.io 


import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import PIL
import json 
import torchvision.models as models
import time
from TREMBA.FCN import *
from TREMBA.utils import *
from TREMBA.dataloader import *
import pandas as pd
import os
from pretrainedmodels import utils as ptm_utils
from pathlib import Path
from utils import * 

from video_caption_pytorch.models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
from video_caption_pytorch.dataloader import VideoDataset
from video_caption_pytorch.process_features import process_batches
from video_caption_pytorch.process_features import create_batches as create_batches_eval
from video_caption_pytorch.models.ConvS2VT import ConvS2VT
from video_caption_pytorch.misc import utils as utils
#%%
def carliniwagner(output, target, k, passed_k):

    #Top value and index of it
    values, indices = torch.topk(output, 1)
    # values, indices = torch.topk(output, 1)

    k = torch.Tensor([k]).cuda()
    for f in range(0, len(output)):

        #If the index of the max matches the target, then max the second highest logit
        if (detach(indices[f])[0] == detach(target[f])):

            #Gets the top two values and their indices
            new_val, new_index = torch.topk(output, 2)

            #Now take second max logit - the logit for target

            # Technically this can just be output[f][detach(new_index[f])[1]] - values[f][0] since it's implied values[f][0] have the target indices.
            measured_value = (output[f][detach(new_index[f])[1]] - output[f][detach(target[f])])

            values[f] = torch.max(measured_value, -k)

            # If -kappa turns out to be bigger, that means we've reached the level of confidence in misclassification we wanted.
            if values[f] == -k:
                passed_k = True
            
            else:
                passed_k = False
            # print(measured_value, values[f], -k)

        else:

            #Take max logit - logit for target
            measured_value = (output[f][detach(indices[f])[0]] - output[f][detach(target[f])])

            values[f] = torch.max(measured_value, -k)
            # print(measured_value, values[f], -k)
    # print(values)

    #This isn't in the Carlini-Wagner attack but I did the mean of the CW result of each frame
    return values.mean(0), passed_k
#%%

def CW_attack(config, delta, model, num_iterations, original, target, optimizer, loss):
    
    torch.cuda.empty_cache()
    # dc = 255.
    dc = config['DC']
    
    # config['c'] = 10000

    passed_k = False
    print("Starting attack")
    
    for i in range(num_iterations):
        
        print("Iteration {}".format(i))
        apply_delta = torch.clamp(delta * 255., min=-dc, max=dc)

        # Original being converted from [0,1] to [0, 255.], so pass in is [0, 255]
        pass_in = torch.clamp(apply_delta + original*255., min=0, max=255.)
        
        tanh_omega = 2 * ((apply_delta + original) / 255.) - 1

        pass_in = 0.5 * (tanh_omega + 1)

        batch = create_batches(pass_in.permute(0,2,3,1), config["DIM"])
        output = model(batch)

        # This gets the actual output of the function
        values, passed_k = carliniwagner(output, target, config["k"], passed_k)

        normterm = 0.5 * ( tanh_omega + 1 ) - (original/255.)

        # normterm = normterm.mean(0).norm()
        normterm = normterm.norm()**2

        cost = normterm + config["c"] * values

        # plt.imshow(pass_in[0].detach().cpu().numpy())
        # plt.show()

        check = []
        for f in range(1):
            check_condition = np.argmax(np.round(detach(output[f]))) == np.round(detach(target[f]))

            print("Checking condition: {} == {} -> {}\nPassed k: {} <= -{} -> {}".format(np.argmax(np.round(detach(output[f]))),
                                                              np.round(detach(target[f])),
                                                              check_condition, detach(values)[0], config['k'], passed_k))

            check.append(check_condition)

        # Attack success
        if (np.array(check).all() == True and passed_k == True):
            print("Early stop at iteration {}".format(i))
            #  when returning pass in is back to [0,1]
            pass_in = torch.clamp(apply_delta + original*255., min=0, max=255.)/255.
            # pass_in = detach(torch.clamp(apply_delta + original*255., min=0, max=255.)[0].permute(1,2,0))/255.
            # plt.imshow(detach(pass_in[0].permute(1,2,0)))
            # plt.show()
            return pass_in, apply_delta, True
            
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Iteration and cost displayed at every step. We apply the perturbation to the original image again to find the adversarial caption.
        print("\nIteration: {}, cost: {}".format(i, cost))
    
    print("Did not converge")
    pass_in = torch.clamp(apply_delta + original*255., min=0, max=255.)/255.
    return pass_in, apply_delta, False
#%%


def CWmain(opt):
    
#     config = json.load(open(opt["config"]))
    config = opt
    
#     import os
#     print(opt['gpuid'])
#     os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(opt['gpuid'])) if type(opt['gpuid']) is list else opt['gpuid']

    ####################################### LOAD CONFIG #######################################################
    # Load the configurations
        
    video_path = config["video_path"]
    
    video_name = config["video_names"]
    conv_model = config["conv_models"]
    eval_model = config["eval_model_name"]
    
    # config["target_classes"] = 207
    target_class = config["target_classes"]
    
    lower_frame = config["lower_frame"]
    upper_frame = config["upper_frame"]
    
    k = config["k"]
    c = config["c"]
    # config['k'] = 40
    # config['c'] = 1

    config["DIM"] = 224
    
    # Append the extension to this depending on whether it is the npy arrays or if it is the video
    save_path = f"{config['save_dir']}/{conv_model}_{target_class}_{video_name}"
    
    csv_save_path = f"{save_path}_CW_{config['victim']}_{config['k']}_{config['c']}_run_summary.csv"
    run_done = Path(csv_save_path)
    if  run_done.is_file():
        print(f"\nRun already found at {csv_save_path}, starting eval")
        # CW_evaluate(config)
        return 
    
    print("Loaded configs")
    
    model_dict = {

        "resnet152": models.resnet152(pretrained=True),
        "densenet121": models.densenet121(pretrained=True),
        "vgg16": models.vgg16(pretrained=True),
        "inceptionv3": models.inception_v3(pretrained=True),
        "googlenet": models.googlenet(pretrained=True),
        "squeezenet": models.squeezenet1_1(pretrained=True),
        "mnasnet": models.mnasnet1_0(pretrained=True),
        "mobilenet_v2": models.mobilenet_v2(pretrained=True),
        # "nasnetalarge": pmodels.nasnetalarge(num_classes=1000, pretrained='imagenet'),
    }
    
    ####################################### PRE-ATTACK CONFIGURATION ##############################################
    
    frames = skvideo.io.vread(f"{video_path}/{video_name}")
    print("Total frames: ", len(frames))

    # Artificially limit the amount of frames
    lower_frame = lower_frame
    upper_frame = upper_frame  # len(frames)
    
    frames = frames[lower_frame:upper_frame]
    

    if config['victim'] == 'eval':
        conv_model = eval_model
    
    conv = model_dict[conv_model]
    conv.eval()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
        
    conv_model = nn.Sequential(
            Normalize(mean, std),
            conv
        )
    
    print("Got model")
    
    tf_img_fn = TransformImage()#ptm_utils.TransformImage(conv)
    load_img_fn = PIL.Image.fromarray
    
    original = original_create_batches(frames_to_do=frames, batch_size=config["batch_size"], tf_img_fn=tf_img_fn,
                                       load_img_fn=load_img_fn)
    
    # fig = plt.figure(figsize=(20,5))
    # # fig.suptitle("Carlini-Wagner Attack Perturbation Examples")

    # np.save("original_frames_train.npy", original[0].numpy())
    # for frame in range(1,5):
    #     fig.add_subplot(1,4,frame)
    #     # plt.title("target: dough")
    #     plt.axis('off')
    #     plt.imshow(torch.Tensor(original[0][frame-1]).permute(1,2,0))
    # fig.subplots_adjust(hspace=0.1, wspace=0.05)
    # plt.savefig("frames_unaltered_cropped.png")

    # Get the original predictions
    with torch.no_grad():
        original_output = conv(original[0]).cuda()#.to(device))
        # print(original_output, original_output.shape)
        print("Original classes: ")
        for f in original_output:
            print(" {} ".format(np.argmax(np.round(f.detach().cpu().numpy()))), end='')
    
    # plt.imshow(original[0].detach().cpu().numpy())
    frames = torch.Tensor(frames).cuda().float()#.unsqueeze(0)



    for_shape = create_batches(frames, config["DIM"])
    adversarial_images = []
    adversarial_frames = []
    
    target = []
    for f in range(1):
        target.append(target_class)

    target = np.array(target, dtype=np.float32)

    # Defining delta, the perturbation itself.
    if torch.cuda.is_available():
        delta = Variable(torch.zeros((frames.shape[0], 3, config["DIM"], config["DIM"])).cuda(),
                         requires_grad=True)
    # 0.005#
    learning_rate = config['learning_rate']
    # learning_rate = config['learning_rate']
    num_iterations = 1000
    # Initializing the Adam optimizer.
    optimizer = optim.Adam([delta],
                           lr=learning_rate,
                           betas=(0.9, 0.999))
    
    loss = nn.CrossEntropyLoss()
    target = torch.Tensor(target).long().cuda()
    ####################################### ATTACK SOURCE #######################################################
    
    conv_model.cuda() 
    pd_array = []
    
    column_names = ["conv_model", "video_name", "victim", "c", "k", "frame_num", "target", "success", "time"]
    tic = time.time()
    # Launching the attack over each frame
    for f in range(len(frames)):
        print("\n-------------------------------\nFrame number: ", f, end='\n')
        
        image = (create_batches(frames[f].unsqueeze(0), config["DIM"]) / 255.)
        image = image.squeeze(0).cuda()
        
        # passin, applydelta, bool
        adv_im, adv_frames, success = CW_attack(config=config, delta=delta[f].unsqueeze(0), model=conv_model, num_iterations=num_iterations, original=image, target=target, optimizer=optimizer, loss=loss)
        # adv_im = torch.clamp(image + adv_im, 0, 255.)
        
        # plt.imshow(detach(adv_im[0].permute(1,2,0)))
        # plt.show()
#         image = (create_batches(frames[f].unsqueeze(0), config["DIM"]) / 255.)
# #         image.cuda()##.to(device)

#         image = image.squeeze(0).cuda()#[0].cuda()
#         adv_frames, success, stats, F.counts = TREMBA_attack(tremba_dict=tremba_dict)

#         adv_im = torch.clamp(image + adv_frames.unsqueeze(0), 0, 1)

#         adv_frames = detach(adv_frames.unsqueeze(0))
        adversarial_images.append(detach(adv_im))#.unsqueeze(0)))
        adversarial_frames.append(detach(adv_frames))
    
        row_array = [config["conv_models"], video_name, config['victim'], config['c'], config['k'], f, target_class, success, ""]
                     
        pd_array.append(row_array)
                     
    print("Attack done")
    toc=  time.time()
    attack_time = toc - tic
    time_row = [config["conv_models"], video_name, config['victim'], config['c'], config['k'], f, target_class, success, attack_time]
    
    pd_array.append(time_row)
    print(f"Attack took: {toc-tic} seconds")

    ####################################### POST-ATTACK ##################################################

    adversarial_frames = np.concatenate(adversarial_frames, axis=0)
    adversarial_images = np.concatenate(adversarial_images, axis=0)

    # plt.imshow(torch.Tensor(adversarial_images[0]).permute(1,2,0))
    # plt.show()

#     plt.imshow(torch.Tensor(adversarial_images[0]).permute(1,2,0))
#     plt.show()
        
    try:
        import os
        print("Creating save path: ", config["save_path"])
        os.mkdir(config["save_path"])
    except:
        print("Save path already exists, skipping creation")
    
    df = pd.DataFrame(pd_array, columns=column_names)
    csv_save_path = f"{save_path}_CW_{config['victim']}_{config['k']}_{config['c']}_run_summary.csv"
    df.to_csv(csv_save_path, index=False)
    print("Pandas results table saved at: ", csv_save_path)
    
    np_save_path = f"{save_path}_CW_{config['victim']}_{config['k']}_{config['c']}_perturbations.npy"
    np.save(np_save_path, adversarial_frames * 255.)
    print(f"Adversarial Perturbations Saved at: {np_save_path}")
    
    np_save_path = f"{save_path}_CW_{config['victim']}_{config['k']}_{config['c']}_adv_images.npy"
    np.save(np_save_path, adversarial_images * 255.)
    print(f"Adv. Images Saved at: {np_save_path}")

    adv_save_path = f"{save_path}_CW_{config['victim']}_{config['k']}_{config['c']}_adv_video.avi"
    print(f"Writing adversarial video to: {adv_save_path}")
    # Writing the adversarial frames to video
    writer = skvideo.io.FFmpegWriter(adv_save_path, outputdict={
        '-c:v': 'huffyuv',  # r210 huffyuv r10k
    })

    for f in adversarial_images:
        writer.writeFrame(f * 255.)

    writer.close()
    print("Finished writing adversarial video.")
    
    # Now eval attack
    CW_evaluate(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--gpuid', nargs='+', type=str, default="0")
    CWmain(parser)