import math
import time


COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

MISSING_VALUE = -1


def print_current_loss(start_time, niter_state, total_niters, losses, epoch=None, sub_epoch=None,
                       inner_iter=None, tf_ratio=None, sl_steps=None):

    def as_minutes(s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def time_since(since, percent):
        now = time.time()
        s = now - since
        es = s / percent
        rs = es - s
        return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

    if epoch is not None:
        print('epoch-inner_iter: %d-%d niter: %d' % (epoch, inner_iter, niter_state), end=" ")

    message = ' %s completed:%3d%%)' % (time_since(start_time, niter_state / total_niters), niter_state / total_niters * 100)


    for k, v in losses.items():
        message += ' %s: %.6f ' % (k, v)
    print(message)


