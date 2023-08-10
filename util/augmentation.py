import numpy as np

class RandomFlip():
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            image = image[:,::-1]
            label = label[:,::-1]
        return image, label


class RandomCrop():
    def __init__(self, crop_rate=0.1, prob=1.0):
        self.crop_rate = crop_rate
        self.prob      = prob

    def __call__(self, image, label):
        if np.random.rand() < self.prob:
            w, h, c = image.shape

            h1 = np.random.randint(0, h*self.crop_rate)
            w1 = np.random.randint(0, w*self.crop_rate)
            h2 = np.random.randint(h-h*self.crop_rate, h+1)
            w2 = np.random.randint(w-w*self.crop_rate, w+1)

            image = image[w1:w2, h1:h2]
            label = label[w1:w2, h1:h2]

        return image, label


class MonoModalRandomCropOut():
    def __init__(self, crop_rate=0.1, prob_rgb=0.5, prob_thermal=0.5):
        #super(RandomCropOut, self).__init__()
        self.crop_rate = crop_rate
        self.prob_rgb = prob_rgb
        self.prob_thermal = prob_thermal

    def __call__(self, image, label):
        seed = np.random.rand()
        if seed <= self.prob_rgb + self.prob_thermal:
            w, h, _ = image.shape

            h1 = np.random.randint(0, h*(1-self.crop_rate))
            w1 = np.random.randint(0, w*(1-self.crop_rate))
            h2 = int(h1 + h*self.crop_rate)
            w2 = int(w1 + w*self.crop_rate)

            if seed <= self.prob_rgb:
                image[w1:w2, h1:h2, :3] = 0.0
            else:
                image[w1:w2, h1:h2, 3:] = 0.0

        return image, label