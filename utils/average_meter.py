class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AverageMeterBatch(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.meter_batch = AverageMeter()
        self.meter_whole = AverageMeter()

    def reset_batch(self):
        self.meter_batch.reset()

    def reset_whole(self):
        self.reset_batch()
        self.meter_whole.reset()

    def update(self, val, n=1):
        self.meter_batch.update(val, n)
        self.meter_whole.update(val, n)

    def avg_all(self):
        return self.meter_whole.avg
    
    def avg_batch(self):
        return self.meter_batch.avg
