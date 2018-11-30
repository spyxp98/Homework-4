# TODO: Сделать градиентный спуск с динамичиским шагом
#       реализовать line_search

import math 
import matplotlib.pyplot as plt
func = lambda x: (x-2)**2

# метод золотого сечения
def line_search(func, x_start):
   traj_x = list()
   iter = 0
   gold_ratio = (1 + math.sqrt(5))/2
   x_end = x_start + 10
   c = x_end - (x_end - x_start)/gold_ratio
   d = x_start + (x_end - x_start)/gold_ratio
   while iter < 100:
      if func(c) < func(d):
         x_end = d
      else:
         x_start = c
      c = x_end - (x_end - x_start) / gold_ratio
      d = x_start + (x_end - x_start) / gold_ratio
      traj_x.append((x_start + x_end)/2)
      iter += 1
   return traj_x


def draw_plot(func, traj_x):
   x = [-5 + 0.1 * i for i in range(100)]
   y = [func(s) for s in x]
   traj_y = [func(s) for s in traj_x]
   plt.plot(x, y)
   plt.scatter(traj_x, traj_y, marker='x', color = 'red')
   plt.scatter(traj_x[-1], func(traj_x[-1]), color = 'green')
   plt.show()
   

if __name__ == '__main__':
   traj_x = line_search(func, 0)
   draw_plot(func, traj_x)