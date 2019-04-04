import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import cv2


class Creation(object):

    def __init__(self, dir_o_pic, frame_num, n_cores):
        super(Creation, self).__init__()
        self._n_cores = n_cores
        self._dir_o_pic = dir_o_pic
        self._pic = cv2.imread(self._dir_o_pic)
        self._pic = cv2.cvtColor(self._pic, cv2.COLOR_BGR2RGB)
        self._gray = cv2.imread(self._dir_o_pic, 0)
        self._bin = cv2.adaptiveThreshold(self._gray, 255, \
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                          cv2.THRESH_BINARY, 101, 15)
        self._worker = self._bin
        self._height = self._worker.shape[0]
        self._width = self._worker.shape[1]
        self._f_num = frame_num

        self._w_arr = np.zeros((self._height,
                                self._width,
                                self._f_num))

        # constants of the Automata
        self._q = np.max(self._worker)
        self._g = 110
        self._k1 = 3
        self._k2 = 3

    def get_pic(self):
        return self._pic

    def get_bin(self):
        return self._bin

    def draw_pic(self):
        plt.figure(1)
        plt.imshow(self._pic)
        plt.figure(2)
        plt.imshow(self._bin, cmap='gray')
        print(self._bin.shape)
        plt.show()

    def draw_worker(self):
        plt.imshow(self._worker, cmap='gray')
        plt.show()

    def _get_cell_state(self, row, col):
        if (0 <= row and row < self._height) and \
           (0 <= col and col < self._width):
           return self._worker[row][col]
        return 0

    # calculate a new state of a cell
    def _new_state(self, row, col):
        current_state = self._get_cell_state(row, col)
        if current_state == 0:
            neighboors = np.zeros(8)
            neighboors[0] = self._get_cell_state(row - 1, col)
            neighboors[1] = self._get_cell_state(row - 1, col - 1)
            neighboors[2] = self._get_cell_state(row, col - 1)
            neighboors[3] = self._get_cell_state(row + 1, col - 1)
            neighboors[4] = self._get_cell_state(row + 1, col)
            neighboors[5] = self._get_cell_state(row + 1, col + 1)
            neighboors[6] = self._get_cell_state(row, col + 1)
            neighboors[7] = self._get_cell_state(row - 1, col + 1)
            # print('Now:', neighboors)
            zeros = np.where(neighboors == 0)[0]
            neighboors = np.delete(neighboors, zeros)  # healthy are not considered
            # print('Then: ', neighboors)
            ills = np.where(neighboors == self._q)[0]  # inf + ill
            # print('Ills: ', ills, ills.shape[0])
            b = ills.shape[0]
            neighboors = np.delete(neighboors, ills)  # only infected remain
            # print('Infs: ', neighboors, neighboors.shape[0])
            a = neighboors.shape[0]
            new_state = int(a / self._k1) + int(b / self._k2)
            if new_state > self._q:
                new_state = self._q
            return new_state
        elif current_state == self._q:
            return 0
        else:
            neighboors = np.zeros(8)
            neighboors[0] = self._get_cell_state(row - 1, col)
            neighboors[1] = self._get_cell_state(row - 1, col - 1)
            neighboors[2] = self._get_cell_state(row, col - 1)
            neighboors[3] = self._get_cell_state(row + 1, col - 1)
            neighboors[4] = self._get_cell_state(row + 1, col)
            neighboors[5] = self._get_cell_state(row + 1, col + 1)
            neighboors[6] = self._get_cell_state(row, col + 1)
            neighboors[7] = self._get_cell_state(row - 1, col + 1)
            s = np.sum(neighboors) + current_state
            # print(neighboors, s)
            zeros = np.where(neighboors == 0)[0]
            neighboors = np.delete(neighboors, zeros)  # healthy are not considered
            ills = np.where(neighboors == self._q)[0]  # inf + ill
            b = ills.shape[0]
            neighboors = np.delete(neighboors, ills)  # only infected remain
            a = neighboors.shape[0]
            new_state = int(s / (a + b + 1)) + self._g
            if new_state > self._q:
                new_state = self._q
            return new_state
        return 0

    def _zhab_rules(self):
        new_worker = np.zeros((self._height, self._width))
        for row in range(self._height):
            for col in range(self._width):
                new_worker[row, col] = self._new_state(row, col)
        self._worker = new_worker

    def run_zhab(self):
        for i in range(self._f_num):
            print('Iter: ', i)
            self._w_arr[:, :, i] = self._worker
            self._zhab_rules()
        np.save('./w_arr.npy', self._w_arr)

    def animate_workers(self):
        fig = plt.figure()
        ims = []
        for i in range(self._f_num):
            im = plt.imshow(self._w_arr[:, :, i], animated=True, cmap='gray')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50,
                                  blit=True, repeat=True,
                                  repeat_delay=1000)
        ani.save('result.gif', dpi=80)
        plt.show()

    def animate_saved(self):
        fig = plt.figure()
        ims_np = np.load('./w_arr.npy')
        print(ims_np.shape)
        ims = []
        for i in range(self._f_num):
            im = plt.imshow(ims_np[:, :, i], animated=True, cmap='gray')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50,
                                        blit=True,
                                        repeat_delay=1000)
        ani.save('result.gif', dpi=80)
        plt.show()





#pic_dir = './pics/chika500x500.png'
#pic_dir = './pics/chika300x300.png'
pic_dir = './pics/chika100x100.png'
#pic_dir = './pics/chika50x50.png'

frames = 30

num_cores = multiprocessing.cpu_count()

if __name__ == '__main__':
    cr1 = Creation(pic_dir, frames, num_cores)
    #cr1.draw_worker()
    cr1.run_zhab()
    #cr1.draw_worker()
    cr1.animate_workers()
    #cr1.animate_saved()
