
import torch
from sklearn.metrics import mean_absolute_error

"""
dim:0 //열
dim:1 //행
dim:2 //depth

[pytorch function]
#tensor.transpose([3,2,1,0]) # dimension 순서 바꿈
#tensor.unsqueeze(!) #remove and add dimension
#tensor.view(tensor.size(0), tensor.size(1), -1) #reshape, -1:1차원 백터
#.t() or .permute() #transpose matrix
#multiply tensor(vector) element- wise : a * b or torch.mul(a, b)
#product matrix : torch.matmul(a,b)
#tensor -> scalar: tensor.item()

[concept]
#euclidian distance: torch.dist(input, other, p-norm)
#average : tensor.mean() or torch.mean(tensor)
#diff(편차) : tensor - average of tensor
#variance(분산): average of (diff)**2
#standard deivation(표준편차): sqrt(variance)
"""

def R_Squared(pred, gt):
    """https://github.com/pytorch/pytorch/issues/1254 참고해서 고치기"""
    pred_diff = pred - pred.mean() #vector - scalar = vector.shape
    gt_diff = gt - gt.mean()
    #sum_mul_diff = torch.sum(pred_diff * gt_diff) #vector * vector = vector
    r_squared = torch.sum(pred_diff * gt_diff) / torch.sqrt(torch.sum(pred_diff**2) * torch.sum(gt_diff**2))
    return r_squared.item()

def MAE(pred, gt):
    #mae = torch.sum(torch.abs(pred-gt)) / pred.view(-1, 1).shape[0]
    mae = mean_absolute_error(gt.cpu(), pred.cpu())
    return mae

def TM(pred, gt):
    """input : vector"""
    euclidian_dist = torch.dist(pred, gt) #scalar
    pred = pred.view(1, -1)
    gt = pred.view(-1, 1)
    product = torch.matmul(pred, gt) #scalar
    tm = product / (product + euclidian_dist ** 2)
    return tm.item()

def SSIM(pred, gt):
    """input : vector"""
    #dynamic range of pred and truth: 20 * log10(max-min)
    #dr = 20*torch.log10(torch.max(pred.max(),gt.max()) - torch.min(pred.min(), gt.min()))
    dr = 1.8
    c1 = (0.01*dr)**2
    c2 = (0.03*dr)**2
    cov = torch.sum((pred - pred.mean()) * (gt - gt.mean())) / pred.view(-1, 1).shape[0]

    ssim = ((2*pred.mean()*gt.mean()+c1) * (2*cov+c2)) / \
        ((pred.mean().pow(2) + gt.mean().pow(2) + c1) * (pred.var() + gt.var() + c2))
    return ssim.item()

if __name__ == "__main__":
    pred = torch.rand(1, 5, 20, 224, 224)
    truth = torch.rand(1, 5, 20, 224, 224)
    #print(pred.shape)

    pred = pred.squeeze(0) #[5, 20, 224, 224]
    truth = truth.squeeze(0)

    r_squared = []
    mae = []
    tm = []
    ssim = []

    channel, depth, width, height = pred.shape
    for c in range(channel):
        pred_c = pred[c, :, :, :]
        truth_c = pred[c, :, :, :]

        pred_v = pred_c.view(1, -1)
        truth_v = truth_c.view(1, -1)

        r_squared.append(R_Squared(pred_v, truth_v))
        mae.append(MAE(pred_v, truth_v))
        tm.append(TM(pred_v, truth_v))
        ssim.append(SSIM(pred_v, truth_v))

    print(r_squared)
    print(mae)
    print(tm)
    print(ssim)


