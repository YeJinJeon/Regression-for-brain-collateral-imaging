import torch
from sklearn.metrics import mean_absolute_error

class RegressionEvaluationMetrics(object):

    @staticmethod
    def r_squared(output, target):
        x = output
        y = target
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost ** 2

    @staticmethod
    def TM(output, target):
        x = output
        y = target
        euclidean_distance = torch.dist(x, y)
        x = x.view(1, -1)
        y = y.view(-1, 1)
        return torch.mm(x, y) / (euclidean_distance ** 2 + torch.mm(x, y))

    @staticmethod
    def ssim(output, target):
        x = output
        y = target
        var_x, mean_x = torch.var_mean(x)
        var_y, mean_y = torch.var_mean(y)
        cov_x_y = torch.sum(torch.mul(x - mean_x, y - mean_y)) / x.view(-1, 1).shape[0]
        c1 = (0.01 * 1.8) ** 2
        c2 = (0.03 * 1.8) ** 2
        return (2 * mean_x * mean_y + c1) * (2 * cov_x_y + c2) / (
                    (mean_x ** 2 + mean_y ** 2 + c1) * (var_x + var_y + c2))

    def mae(self, output, target):
        output = output.view(1, -1)
        target = target.view(1, -1)
        mae = mean_absolute_error(target.cpu(), output.cpu())
        #pixel_count = masks.sum(dim=(1,2,3))
        #mae = torch.sum(torch.abs(output - target)) / pixel_count
        return mae
