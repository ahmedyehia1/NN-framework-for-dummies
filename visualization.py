import matplotlib.pyplot as plt
from matplotlib import style

class visualization:
    def __init__(self,name="the graph",line1="loss",line2="percision"):
        self.value1 = []
        self.value2 = []
        self.value3 = []

        self.line1 = line1
        self.line2 = line2


        style.use('fivethirtyeight')
        # plt.xlabel('iteration')

        fig = plt.figure(name)
        self.ax1 = fig.add_subplot(1, 1, 1)
        plt.ion()
        plt.show()
        # plt.show(block = False)

    def add_point_to_graph(self,new_value3):
        self.value3.append(new_value3)


        if len(self.value3) > 1 :
            self.ax1.clear()
            # self.ax1.plot(self.counter, self.value , c='black')
            self.ax1.plot( self.value3 )
            # plt.draw()
            plt.pause(.001)

    def add_two_points_to_graph(self,new_value1,new_value2):
        self.value1.append(new_value1)
        self.value2.append(new_value2)


        if len(self.value1) > 1 :
            self.ax1.clear()
            self.ax1.plot( self.value1 ,label=self.line1)
            self.ax1.plot(self.value2,label=self.line2)
            plt.legend()
            # plt.draw()
            plt.pause(.001)

  #default values
graph_title = "the graph"
first_line = "loss"
seconde_line = "percision"

p1 = visualization(graph_title,first_line,seconde_line)
# p1 = visualization()  # optional attributes

p1.add_point_to_graph(5) # add point to the line to draw (one line) # will work after 2nd point
p1.add_two_points_to_graph(10,4) # draw two lines

