# TODO: реализовать метод градиентного спуска и метод 
#       сопряженных градиентов.
# тестим на функции f(x, y) = cos^2(x)*sin(y)
# grad f = {-2 cos(x)*sin(x)*sin(y), cos^2(x)*cos(y)}
import math
import matplotlib.pyplot as plt
import numpy as np

func4 = lambda x, y: x**2 + y**2
func4_grad_x = lambda x, y: 2*x
func4_grad_y = lambda x, y: 2*y

func3 = lambda x, y: (x-y)**2
func3_grad_x = lambda x, y: 2*(x-y)
func3_grad_y = lambda x, y: -2*(x-y)

func2 = lambda x, y: math.cos(math.pi * x) * (y-1)**2
func2_grad_x = lambda x, y: -math.pi * math.sin(math.pi * x) * (y-1)**2
func2_grad_y = lambda x, y: math.cos(math.pi * x) * 2 * (y-1)

func1 = lambda x, y: (math.cos(x))**2 * math.sin(y)
func1_grad_x = lambda x, y: -2*math.cos(x)*math.sin(x)*math.sin(y)
func1_grad_y = lambda x, y: (math.cos(x))**2 * math.cos(y)

dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def line_search(func, x_start, y_start, dir_x, dir_y):
   gold_ratio = (3 - math.sqrt(5))/2
   eps = 0.01
   x_end = x_start + 10 * dir_x
   y_end = y_start + 10 * dir_y
   x_middle = x_start + (x_end - x_start) * gold_ratio
   y_middle = y_start + (y_end - y_start) * gold_ratio
   while dist(x_start, y_start, x_end, y_end) > eps:
      d_x = x_start - (x_end - x_start) * gold_ratio
      d_y = y_start - (y_end - y_start) * gold_ratio
      if func(d_x, d_y) < func(x_middle, y_middle):
         x_start, x_middle = x_middle, d_x
         y_start, y_middle = y_middle, d_y
      elif func(d_x, d_y) > func(x_middle, y_middle):
         x_end, x_start = x_start, d_x
         y_end, y_start = y_start, d_y
   return x_middle, y_middle


def conjugate_gradient(x_start, y_start, eps, func, grad_x, grad_y):
   max_iterations = 1000
   iter = 0

      

def line_search(func, x_start, y_start, dir_x, dir_y):
   # dir = grad_y(x_start, y_start) / grad_x(x_start, y_start)
   step = 0.001
   x = x_start + step * dir_x
   y = y_start + step * dir_y
   while func(x, y) < func(x_start, y_start):
      x += step * dir_x
      y += step * dir_y

   return dist(x, y, x_start, y_start)

def conjugate_gradient(func, grad_x, grad_y):
   traj_x = list()
   traj_y = list()
   max_iterations = 1000
   precision = 0.001
   x_start, y_start = -4, 4
   iter = 0
   step = 10
   new_dir_x = -grad_x(x_start, y_start)
   new_dir_y = -grad_x(x_start, y_start)
   new_x = x_start
   new_y = y_start
   prev_x = new_x
   prev_y = new_y
   while step > precision and iter < max_iterations:
      chng = (grad_x(new_x, new_y) * (grad_x(new_x, new_y) - grad_x(prev_x, prev_y)) + grad_y(new_x, new_y) * (grad_y(new_x, new_y) - grad_y(prev_x, prev_y)))/(grad_x(prev_x, prev_y) ** 2 + grad_y(prev_x, prev_y) ** 2)
      prev_dir_x = new_dir_x
      prev_dir_y = new_dir_y
      new_dir_x = -grad_x(new_x, new_y) + chng * prev_dir_x
      new_dir_y = -grad_y(new_x, new_y) + chng * prev_dir_y
      dir = new_dir_y / new_dir_x
      step = line_search(func, new_x, new_y, new_dir_x, new_dir_y)
      prev_x = new_x
      prev_y = new_y
      new_x += step * new_dir_x
      new_y += step * new_dir_y
      traj_x.append(new_x)
      traj_y.append(new_y)
      iter += 1

   print(f'Minimum at x = {new_x}, y = {new_y}. \n Number of iterations - {iter}')
   return traj_x, traj_y

def draw_plot(func, traj_x, traj_y):
   x = [-5 + 0.1 * i for i in range(100)]
   y = [-5 + 0.1 * i for i in range(100)]
   z = list()
   for i in range(100):
      z.append([])
      for j in range(100):
         z[i].append(func(x[i], y[j]))
   points_x = [traj_x[0], traj_x[-1]]
   points_y = [traj_y[0], traj_y[-1]]
   h = plt.contourf(x,y,z)
   plt.contour(h, colors = 'k')
   plt.colorbar(h)
   traj = plt.plot(traj_x, traj_y, 'k--')
   plt.scatter(points_x, points_y, color = 'red', marker = 'x')
   plt.show()

if __name__ == '__main__':
   traj_x, traj_y = gradient_descent(func4_grad_x, func4_grad_y)
   # print(traj_x)
   draw_plot(func4, traj_x, traj_y)
