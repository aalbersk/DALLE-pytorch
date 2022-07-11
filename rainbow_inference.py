from tqdm import tqdm
from dalle_pytorch import DiscreteVAE, DALLE
from einops import repeat
import glob
import itertools
import os
import numpy as np
import time
import torch

# vision imports
from io import BytesIO
from PIL import Image
from torchvision.utils import save_image

INFERENCE_CAPTION = 'big filled rainbow square'
NUM_IMAGES = 512
BATCH_SIZE = 128
IF_SAVED_IMAGES = False

if __name__ == '__main__':
    print('You need to run examples/rainbow_dalle.ipynb to train the models and receive all available caption name.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    vae_model_file = 'examples/data/rainbow_vae.model'
    vae = DiscreteVAE(
        image_size = 32,
        num_layers = 3,          # number of downsamples - ex. 256 / (2 ** 3) = (32 x 32 feature map)
        num_tokens = 256,       # number of visual tokens. in the paper, they used 8192, but could be smaller for downsized projects
        codebook_dim = 512,      # codebook dimension
        hidden_dim = 64,         # hidden dimension
        num_resnet_blocks = 2,   # number of resnet blocks
        temperature = 0.9,       # gumbel softmax temperature, the lower this is, the harder the discretization
        straight_through = False # straight-through for gumbel softmax. unclear if it is better one way or the other
    ).to(device)
    vae.load_state_dict(torch.load(vae_model_file))

    dalle_model_file = 'examples/data/rainbow_dalle.model'
    dalle = DALLE(
        dim = 1024,
        vae = vae,                  # automatically infer (1) image sequence length and (2) number of image tokens
        num_text_tokens = 32,    # vocab size for text
        text_seq_len = 7,         # text sequence length
        depth = 12,                 # should aim to be 64
        heads = 16,                 # attention heads
        dim_head = 64,              # attention head dimension
        attn_dropout = 0.1,         # attention dropout
        ff_dropout = 0.1            # feedforward dropout
    ).to(device)
    dalle.load_state_dict(torch.load(dalle_model_file))


    inference_caption_words = INFERENCE_CAPTION.split()

    captions = []
    for fn in glob.glob('examples/data/rainbow/*.png'):
        captions.append(os.path.basename(fn).replace('.png', '').split('_'))

    all_words = list(sorted(frozenset(list(itertools.chain.from_iterable(captions)))))
    word_tokens = dict(zip(all_words, range(1, len(all_words) + 1)))
    caption_tokens = [[word_tokens[w] for w in c] for c in captions]

    inference_caption_token = [word_tokens[w] for w in inference_caption_words]

    longest_caption = max(len(c) for c in captions)
    inference_caption_array = np.zeros((1, longest_caption), dtype=np.int64)
    inference_caption_array[0, :len(inference_caption_token)] = inference_caption_token

    print(f'{INFERENCE_CAPTION} : {inference_caption_words} : {inference_caption_token} : {inference_caption_array}')

    inference_caption_array = torch.from_numpy(inference_caption_array).to(device)


    inference_caption_array = repeat(inference_caption_array, '() n -> b n', b = NUM_IMAGES)

    generated_images = []
    latency_list = []
    for x in range(int(NUM_IMAGES/BATCH_SIZE)):
        tick = time.perf_counter()

    with torch.no_grad():
        for text_chunk in tqdm(inference_caption_array.split(BATCH_SIZE), desc = f'generating images for - {INFERENCE_CAPTION}'):
            generated = dalle.generate_images(inference_caption_array, temperature=0.00001)
            generated_images.append(generated)

    latency_time = time.perf_counter() - tick
    latency_list.append(latency_time)

    P50_latency = np.percentile(np.array(latency_list), 50) * 1000
    P90_latency = np.percentile(np.array(latency_list), 90) * 1000
    avg_latency = np.average(np.array(latency_list))
    throughput = BATCH_SIZE / avg_latency

    print(f'Batch size: {BATCH_SIZE}')
    print(f'Number of images: {NUM_IMAGES}')
    print(f'P50 latency: {round(P50_latency, 4)} sec')
    print(f'P90 latency: {round(P90_latency, 4)} sec')
    print(f'Average latency: {round(avg_latency, 4)} sec')
    print(f'Throughput: {round(throughput, 4)} sec')

    if IF_SAVED_IMAGES:
        generated_images = torch.cat(generated_images)

        # save all images
        outputs_dir = 'testing'
        outputs_dir = os.path.join(outputs_dir, INFERENCE_CAPTION.replace(' ', '_')[:(100)])
        try:
            os.makedirs(outputs_dir)
        except FileExistsError:
            print(f'Directory  {outputs_dir} already exists')

        for i, image in tqdm(enumerate(generated_images), desc = 'saving images'):
            np_image = np.moveaxis(image.cpu().numpy(), 0, -1)
            formatted = (np_image * 255).astype('uint8')
            img = Image.fromarray(formatted)

            buffered = BytesIO()
            img.save(buffered, format='JPEG')
            save_image(image, os.path.join(outputs_dir, f'{i}.jpg'), normalize=True)

        print(f'created {NUM_IMAGES} images at "{str(outputs_dir)}"')
