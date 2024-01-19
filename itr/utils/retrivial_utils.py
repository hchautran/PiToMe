import torch
import torch.nn.functional as F
import numpy as np



def evaluate_recall(sims_t2i:torch.Tensor, mode='val'):

    sims_t2i = sims_t2i.numpy()
    recall_t2i = t2i(sims_t2i)
    recall_i2t = i2t(sims_t2i.T)
    
    r1i, r5i, r10i = recall_i2t
    r1t, r5t, r10t = recall_t2i
    output = {
        f'{mode}/r1_i2t': r1i,
        f'{mode}/r5_i2t': r5i,
        f'{mode}/r10_i2t': r10i,
        f'{mode}/r_i2t': r1i + r5i + r10i,
        f'{mode}/r1_t2i': r1t,
        f'{mode}/r5_t2i': r5t,
        f'{mode}/r10_t2i': r10t,
        f'{mode}/r_t2i': r1t + r5t + r10t,
        f'{mode}/r_all': r1t + r5t + r10t + r1i + r5i + r10i,
    }
    return output 

def itm_t2i(itm_t2i_logits:torch.Tensor, targets:torch.Tensor):
    n_text = len(itm_t2i_logits)
    itm_t2i_logits  = itm_t2i_logits.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    ranks = np.zeros(n_text)
    
    for index in range(int(n_text/5)):
        for i in range(5):
            if index in targets[5 * index + i]:
                inds = np.argsort(itm_t2i_logits[5 * index + i])[::-1]
                target = np.take(targets[5 * index + i], inds)
                ranks[5 * index + i] = np.where(target == index)[0][0]
            else:
                ranks[5 * index + i] = 1e20 
    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10


def i2t(sims_i2t):
    # sims (n_imgs, n_caps)
    n_imgs, _ = sims_i2t.shape
    ranks = np.zeros(n_imgs)
    for index in range(n_imgs):
        inds = np.argsort(sims_i2t[index])[::-1]
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10



def t2i(sims_t2i):
    # sims (n_caps, n_imgs)
    _, n_imgs = sims_t2i.shape
    ranks = np.zeros(5*n_imgs)
    # --> (5N(caption), N(image))
    for index in range(n_imgs):
        for i in range(5):
            inds = np.argsort(sims_t2i[5 * index + i])[::-1]
            # print(index)
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            # print(np.where(inds == index))


    # Compute metrics
    r1 = 1.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 1.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 1.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return r1, r5, r10

def report_metrics(scores_i2t:torch.Tensor, scores_t2i:torch.Tensor, txt2img, img2txt, mode='val'):
    scores_i2t = scores_i2t.numpy()
    scores_t2i = scores_t2i.numpy()

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    output = {
        f'{mode}/r1_i2t': tr1,
        f'{mode}/r5_i2t': tr5,
        f'{mode}/r10_i2t': tr10,
        f'{mode}/r_i2t': tr1 + tr5 + tr10,
        f'{mode}/r1_t2i': ir1,
        f'{mode}/r5_t2i': ir5,
        f'{mode}/r10_t2i': ir10,
        f'{mode}/r_t2i': ir1 + ir5 + ir10,
        f'{mode}/r_all': tr1 + tr5 + tr10 + ir1 + ir5 + ir10,
    }
    return output

