import pyglet
import numpy as np

class Lattice:

    def __init__(self, window_width, window_height, cell_size):
        self.grid_width = int(window_width / cell_size)
        self.grid_height = int(window_height / cell_size)
        self.cell_size = cell_size
        self.configurations = self.get_sites()
        self.config_num = 0

    def get_sites(self):
        with open('ising_ca.txt', 'r') as file:
            configurations = []
            line = file.readline()
            while line:
                configuration = line.strip().split(' ')
                configurations.append(configuration)
                line = file.readline()
        return configurations

    def draw(self):
        try:
            configuration = self.configurations[self.config_num]
        except IndexError:
            print('The simulation is over!')
            quit()
        L = int(np.sqrt(len(configuration)))
        for i in range(len(configuration)):
            row = int(i/L)
            col = i % L
            if int(configuration[i]) == 1:
                square_coords = (row*self.cell_size, col*self.cell_size,
                                 row*self.cell_size, col*self.cell_size + self.cell_size,
                                 row*self.cell_size + self.cell_size, col*self.cell_size,
                                 row*self.cell_size+self.cell_size, col*self.cell_size+self.cell_size)
                pyglet.graphics.draw_indexed(4, pyglet.gl.GL_TRIANGLES, [0, 1, 2, 1, 2, 3], ('v2i', square_coords))

class Window(pyglet.window.Window):

    def __init__(self):
        super().__init__(600, 600)
        self.lattice = Lattice(self.get_size()[0], self.get_size()[1], int(self.get_size()[0]/50))
        pyglet.clock.schedule_interval(self.update, 1.0/25.0)

    def on_draw(self):
        self.clear()
        self.lattice.draw()

    def update(self, dt):
        self.lattice.config_num += 1

if __name__ == '__main__':
    window = Window()
    pyglet.app.run()