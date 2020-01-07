import math
from urllib.request import urlretrieve
import torch
from PIL import Image
from tqdm import tqdm


class Warp(object):
    """ Warp any image to a predefined image size"""

    def __init__(self, size, interpolation=Image.ANTIALIAS):
        self.height = int(size[1])
        self.width = int(size[0])
        self.interpolation = interpolation

    def __call__(self, img):
        return img.resize((self.height, self.width), self.interpolation)

    def __str__(self):
        return self.__class__.__name__ + ' (size={size}, interpolation={interpolation})'.format(size=self.size,
                                                                                                interpolation=self.interpolation)

class AveragePrecisionMeter(object):
    """ 
    The AveragePrecisionMeter measures the average precision per class as well as the F1-scores.
    The AveragePrecisionMeter is designed to operate on `NxK` Tensors `output` and
    `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
    contains model output scores for `N` examples and `K` classes that ought to
    be higher when the model is more convinced that the example should be
    positively labeled, and smaller when the model believes the example should
    be negatively labeled (for instance, the output of a sigmoid function); (2)
    the `target` contains only values 0 (for negative examples) and 1
    (for positive examples); and (3) the `weight` ( > 0) represents weight for
    each sample.
    """

    def __init__(self):
        super(AveragePrecisionMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the meter with empty member variables"""
        self.predictions = torch.FloatTensor(torch.FloatStorage())
        self.scores = torch.FloatTensor(torch.FloatStorage())
        self.targets = torch.LongTensor(torch.LongStorage())
        self.weights = torch.FloatTensor(torch.FloatStorage())

    def _2d_to_onehot(self, x):
        """ Converts 2d array to 3d one hot matrix"""
        return (torch.arange(10).cuda().long() == x[...,None].long()).int()

    def flatten(self, x):
        return x.reshape(-1, x.size(-1))

    def add(self, thresh_pred, output, target, weight=None):
        """ Add a new observation
            Args:
                thresh_pred (Tensor): NxHxW tensor that for each of the N 
                    examples indicates the predicted class at each pixel location
                    according to the model.
                output (Tensor): NxHxWxK tensor that for each of the N examples
                    indicates the probability of the each pixel of example belonging to each of
                    the K classes, according to the model. The probabilities should
                    sum to one over all classes in 4th dimension.
                target (Tensor): binary NxHxW tensort that encodes which of the K
                    classes is associated with the every pixel in N-th input.
                weight (optional, Tensor): Nx1 tensor representing the weight for
                    each example (each weight > 0)
        """


        thresh_pred = self.flatten(self._2d_to_onehot(thresh_pred))

        output = self.flatten(output.permute(0,2,3,1))

        target = self.flatten(self._2d_to_onehot(target.squeeze(1)))

        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        if weight is not None:
            if not torch.is_tensor(weight):
                weight = torch.from_numpy(weight)
            weight = weight.squeeze()
        if output.dim() == 1:
            output = output.view(-1, 1)
        else:
            assert output.dim() == 2, \
                'wrong output size (should be 1D or 2D with one column \
                per class)'
        if target.dim() == 1:
            target = target.view(-1, 1)
        else:
            assert target.dim() == 2, \
                'wrong target size (should be 1D or 2D with one column \
                per class)'
        if weight is not None:
            assert weight.dim() == 1, 'Weight dimension should be 1'
            assert weight.numel() == target.size(0), \
                'Weight dimension 1 should be the same as that of target'
            assert torch.min(weight) >= 0, 'Weight should be non-negative only'
        # print(target.cpu().numpy())
        assert torch.equal(target**2, target), \
            'targets should be binary (0 or 1)'
        if self.scores.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.scores.storage().size() < self.scores.numel() + output.numel():
            new_size = math.ceil(self.scores.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.scores.storage().resize_(int(new_size + output.numel()))
            self.targets.storage().resize_(int(new_size + output.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + output.size(0)))

        # store scores and targets
        offset = self.scores.size(0) if self.scores.dim() > 0 else 0
        self.scores.resize_(offset + output.size(0), output.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.scores.narrow(0, offset, output.size(0)).copy_(output)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

        if weight is not None:
            self.weights.resize_(offset + weight.size(0))
            self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

        ##################################################################################

        if not torch.is_tensor(thresh_pred):
            thresh_pred = torch.from_numpy(thresh_pred)
        
        if thresh_pred.dim() == 1:
            thresh_pred = thresh_pred.view(-1, 1)
        else:
            assert thresh_pred.dim() == 2, \
                'wrong prediction size (should be 1D or 2D with one column \
                per class)'

        if self.predictions.numel() > 0:
            assert target.size(1) == self.targets.size(1), \
                'dimensions for output should match previously added examples.'

        # make sure storage is of sufficient size
        if self.predictions.storage().size() < self.predictions.numel() + thresh_pred.numel():
            new_size = math.ceil(self.predictions.storage().size() * 1.5)
            new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
            self.predictions.storage().resize_(int(new_size + thresh_pred.numel()))
            self.targets.storage().resize_(int(new_size + thresh_pred.numel()))
            if weight is not None:
                self.weights.storage().resize_(int(new_weight_size + thresh_pred.size(0)))

        # store scores and targets
        offset = self.predictions.size(0) if self.predictions.dim() > 0 else 0
        self.predictions.resize_(offset + thresh_pred.size(0), thresh_pred.size(1))
        self.targets.resize_(offset + target.size(0), target.size(1))
        self.predictions.narrow(0, offset, thresh_pred.size(0)).copy_(thresh_pred)
        self.targets.narrow(0, offset, target.size(0)).copy_(target)

    def value(self):
        """
            Returns the model's average precision for each class
            
            Return:
                ap (FloatTensor): 1xK tensor, with avg precision for each class k
        """

        if self.scores.numel() == 0:
            return 0
        ap = torch.zeros(self.scores.size(1))
        if hasattr(torch, "arange"):
            rg = torch.arange(1, self.scores.size(0) + 1).float()
        else:
            rg = torch.range(1, self.scores.size(0)).float()
        if self.weights.numel() > 0:
            weight = self.weights.new(self.weights.size())
            weighted_truth = self.weights.new(self.weights.size())

        # compute average precision for each class
        for k in range(self.scores.size(1)):
            # sort scores
            scores = self.scores[:, k]
            targets = self.targets[:, k]
            _, sortind = torch.sort(scores, 0, True)
            truth = targets[sortind]
            if self.weights.numel() > 0:
                weight = self.weights[sortind]
                weighted_truth = truth.float() * weight
                rg = weight.cumsum(0)

            # compute true positive sums
            if self.weights.numel() > 0:
                tp = weighted_truth.cumsum(0)
            else:
                tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)

            # compute average precision
            ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)
        return ap

    def value_metrics(self):
        """
            Returns the model's TPs, TNs, FPs and FNs for each class
            Return:
                TP (FloatTensor): 1xK tensor, with True Positives for each class k
                FP (FloatTensor): 1xK tensor, with False Positives for each class k
                TN (FloatTensor): 1xK tensor, with True Negatives for each class k
                FN (FloatTensor): 1xK tensor, with False Negatives for each class k
        """

        if self.predictions.numel() == 0:
            return 0

        print(self.predictions.size())

        TP = torch.zeros(self.predictions.size(1))
        FP = torch.zeros(self.predictions.size(1))
        TN = torch.zeros(self.predictions.size(1))
        FN = torch.zeros(self.predictions.size(1))


        precision = torch.zeros(self.predictions.size(1))
        recall = torch.zeros(self.predictions.size(1))
        f1 = torch.zeros(self.predictions.size(1))


        for k in range(self.predictions.size(1)):
            
            predictions = (self.predictions[:, k]).byte()
            targets = (self.targets[:, k]).byte()

            one_minus_pred = (1 - predictions)
            one_minus_tar = (1 - targets)

            TP[k] = torch.mul(predictions, targets).sum() #TP
            FP[k] = torch.mul(predictions, one_minus_tar).sum() #FP
            TN[k] = torch.mul(one_minus_pred, one_minus_tar).sum() #TN
            FN[k] = torch.mul(one_minus_pred, targets).sum() #FN


        cm = torch.einsum('bi,bj->bij', self.targets.float(), self.predictions.float()).sum(0)

        sum_over_row = cm.sum(0)

        sum_over_col = cm.sum(1)

        cm_diag = torch.diag(cm)

        denominator = sum_over_row + sum_over_col - cm_diag

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = torch.where((denominator > 0), denominator, torch.ones_like(denominator))

        # Calculating Intersection Over Union
        iou = (cm_diag / denominator)

        # Freq weight IoU
        fiou = torch.mul(sum_over_row, iou).sum() / cm.sum().float()

        print("IOU = ", iou.mean())
        print("F_IOU = ", fiou)
        print("Accuracy = ", (cm_diag.sum() / cm.sum()))

        return TP, FP, TN, FN