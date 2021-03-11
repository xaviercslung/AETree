import torch 
import pickle 
import numpy as np 
from StructuralLosses.match_cost import match_cost
from StructuralLosses.nn_distance import nn_distance
import argparse 


parser = argparse.ArgumentParser()
parser.add_argument('--Training_dataroot', default="../box_set_data_train.pickle",required=False, help='path to training dataset')
parser.add_argument('--Testing_dataroot', default="../box_set_data_test.pickle",required=False, help='path to testing dataset')

opt = parser.parse_args()

# In[2]:

def distChamferCUDA(x, y):
    return nn_distance(x, y)


def emd_approx(sample, ref):
    B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
    assert N == N_ref, "Not sure what would EMD do in this case"
    emd = match_cost(sample, ref)  # (B,)
    emd_norm = emd / float(N)  # (B,)
    return emd_norm

def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]


# In[3]:

def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'lgan_mmd': mmd,
        'lgan_cov': cov,
        'lgan_mmd_smp': mmd_smp,
    }


def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]

        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]
            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.unsqueeze(0).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            if accelerated_cd:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)

            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            emd_batch = emd_approx(sample_batch_exp, ref_batch)
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


# In[4]:


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=False):
    results = {}

    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })
    return results
      

data_folder =opt.Training_dataroot
with open(data_folder, 'rb') as filename:
    box_set_train = pickle.load(filename)
data_folder =opt.Testing_dataroot
with open(data_folder, 'rb') as filename:
    box_set_test = pickle.load(filename)


result=compute_all_metrics(torch.FloatTensor(box_set_train).cuda(),torch.FloatTensor(box_set_test).cuda(),100,accelerated_cd=False)
print(result)





