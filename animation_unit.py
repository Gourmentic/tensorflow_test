import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np



class animation_maker():
    '''
    This is a class to draw a motion curve.

    Paraments:
    ---------

    motion_data: tuple(list,list), contain the values of x_axis and y_axis
    animation_type: string, 'point' or 'curve'
    xlable: string, label of x_axis
    ylable: string, label of y_axis
    basic_data: tuple(list,list), one chance to draw something not change
    basic_type: string, refer to plt.plot line type and color setting
    save_path: save the motion as mp4 or gif
    '''

    def __init__(self,
                 motion_date,
                 motion_type='ro',
                 animation_type='point',
                 plt_shown=True,
                 xlabel=None,
                 ylabel=None,
                 basic_data=([],[]),
                 basic_type='b--',
                 save_path=None):

        self.fig, self.ax = plt.subplots()
        self.motion_x, self.motion_y = motion_date
        self.basic_x, self.basic_y = basic_data
        self.ln = self.ax.plot([], [], motion_type, animated=True)
        self.ln_init = self.ax.plot(self.basic_x, self.basic_y, basic_type)
        self.animation_type = animation_type
        self.plt_shown = plt_shown
        self.save_path = save_path

        if len(self.basic_x) != 0:

            self.ax.set_xlim(min(self.basic_x), max(self.basic_x))
            self.ax.set_ylim(min(self.basic_y), max(self.basic_y))
        else:
            self.ax.set_xlim(min(self.motion_x), max(self.motion_x))
            self.ax.set_ylim(min(self.motion_y), max(self.motion_y))

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

    def update(self, frame):

        if self.animation_type == 'point':
            self.ln[0].set_data(self.motion_x[frame],self.motion_y[frame])

        return self.ln

    def init(self):

        return self.ln_init

    def draw_animation(self):

        ani = FuncAnimation(self.fig, self.update, frames=np.arange(len(self.motion_x)),
                            init_func=self.init, blit=True)

        if self.plt_shown:
            plt.show()

        if self.save_path is not None:

            if self.save_path[-3:] == 'mp4 ':
                ani.save(save_path, writter == 'ImageMagick')

            elif self.save_path[-3:] == 'gif':
                ani.save(save_paht, writter == 'ImageMagick')

            else:
                print('please check the extension name of file you want to save')


if __name__ == "__main__":

    am = animation_maker(([1,2,3,4],[1,2,3,4]),basic_data=([1,2,3,4],[1,2,3,4]))

    am.draw_animation()
