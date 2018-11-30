# TODO: реализовать метод градиентного спуска и метод 
#       сопряженных градиентов.
# тестим на функции f(x, y) = cos^2(x)*sin(y)
# grad f = {-2 cos(x)*sin(x)*sin(y), cos^2(x)*cos(y)}
import math
import matplotlib.pyplot as plt
import numpy as np

func4 = lambda x, y: (x/10)**2 + (y/5)**2
func4_grad_x = lambda x, y: 2*x/100
func4_grad_y = lambda x, y: 2*y/25

func3 = lambda x, y: (x-y)**2
func3_grad_x = lambda x, y: 2*(x-y)
func3_grad_y = lambda x, y: -2*(x-y)

func2 = lambda x, y: math.cos(math.pi * x) * (y-1)**2
func2_grad_x = lambda x, y: -math.pi * math.sin(math.pi * x) * (y-1)**2
func2_grad_y = lambda x, y: math.cos(math.pi * x) * 2 * (y-1)

func1 = lambda x, y: (math.cos(x))**2 * math.sin(y)
func1_grad_x = lambda x, y: -2*math.cos(x)*math.sin(x)*math.sin(y)
func1_grad_y = lambda x, y: (math.cos(x))**2 * math.cos(y)

rosen = lambda x, y: (1-x)**2 + 100 * (y - x**2)**2
rosen_grad_x = lambda x, y: 2 * (x - 1) + 200 * (y - x**2) * (-2*x)
rosen_grad_y = lambda x, y: 200 * (y - x ** 2) 

dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def gradient_descent(func, grad_x, grad_y):
   traj_x = list()
   traj_y = list()
   max_iterations = 1000
   step = 0.5
   eps = 0.001
   x_new = 4
   y_new = 4
   last_step = 10
   iter = 0
   while iter < max_iterations and last_step > eps:
      x_prev = x_new
      y_prev = y_new
      x_new -= step * grad_x(x_prev, y_prev)
      y_new -= step * grad_y(x_prev, y_prev)
      traj_x.append(x_new)
      traj_y.append(y_new)
      last_step = dist(x_prev, y_prev, x_new, y_new)
      iter += 1
   print(f'Minimum at x = {traj_x[-1]}, y = {traj_y[-1]}. Iter - {iter}')
   return traj_x, traj_y

def line_search(func, x_start, y_start, dir_x, dir_y):
   traj_x = list()
   traj_y = list()
   iter = 0
   gold_ratio = (1 + math.sqrt(5))/2
   x_end = x_start + 100 * dir_x
   y_end = y_start + 100 * dir_y
   c_x = x_end - (x_end - x_start)/gold_ratio
   d_x = x_start + (x_end - x_start)/gold_ratio
   c_y = y_end - (y_end - y_start)/gold_ratio
   d_y = y_start + (y_end - y_start)/gold_ratio
   while iter < 100:
      if func(c_x, c_y) < func(d_x, d_y):
         x_end = d_x
         y_end = d_y
      else:
         x_start = c_x
         y_start = c_y
      c_x = x_end - (x_end - x_start)/gold_ratio
      d_x = x_start + (x_end - x_start)/gold_ratio
      c_y = y_end - (y_end - y_start)/gold_ratio
      d_y = y_start + (y_end - y_start)/gold_ratio
      traj_x.append((x_start + x_end)/2)
      traj_y.append((y_start + y_end)/2)
      iter += 1
   return traj_x, traj_y

def conjugate_gradient(func, grad_x, grad_y):
   # инициализация переменных
   x_start, y_start = -10, 4
   max_iterations = 1000
   iter = 0
   last_step = 10
   traj_x = list()
   traj_y = list()
   eps = 0.0001
   x_new = x_start
   y_new = y_start
   dir_x_new = -1 * grad_x(x_start, y_start)
   dir_y_new = -1 * grad_y(x_start, y_start)
   while last_step > eps and iter < max_iterations:
      dir_x_prev = dir_x_new
      dir_y_prev = dir_y_new
      x_prev = x_new
      y_prev = y_new
      temp_x, temp_y = line_search(func, x_prev, y_prev, dir_x_prev, dir_y_prev)
      x_new, y_new = temp_x[-1], temp_y[-1]
      traj_x.append(x_new)
      traj_y.append(y_new)
      try:
         beta = (((grad_x(x_new, y_new))**2 + (grad_y(x_new, y_new))**2) / 
                   ((grad_x(x_prev, y_prev))**2 + (grad_y(x_prev, y_prev))**2))
         # beta = ((grad_x(x_new, y_new) * (grad_x(x_new, y_new) - grad_x(x_prev, y_prev)) + 
               # grad_y(x_new, y_new) * (grad_y(x_new, y_new) - grad_y(x_prev, y_prev))) / 
               # (dir_x_prev * (grad_x(x_new, y_new) - grad_x(x_prev, y_prev)) + 
               #  dir_y_prev * (grad_y(x_new, y_new) - grad_y(x_prev, y_prev))))
      except ZeroDivisionError:
         break
      dir_x_new = - 1 * grad_x(x_new, y_new) + dir_x_prev * beta
      dir_y_new = - 1 * grad_y(x_new, y_new) + dir_y_prev * beta
      last_step = dist(x_new, y_new, x_prev, y_prev)
      iter += 1
   print(f'Minimum at x = {traj_x[-1]}, y = {traj_y[-1]}. Iter - {iter}')
   return traj_x, traj_y

def draw_plot(func, traj_x, traj_y):
   x = [-10 + 0.1 * i for i in range(200)]
   y = [-10 + 0.1 * i for i in range(200)]
   z = list()
   for i in range(200):
      z.append([])
      for j in range(200):
         z[i].append(func(x[i], y[j]))
   points_x = [traj_x[0], traj_x[-1]]
   points_y = [traj_y[0], traj_y[-1]]
   h = plt.contourf(x,y,z)
   plt.contour(h, colors = 'k')
   plt.colorbar(h)
   traj = plt.scatter(traj_x, traj_y, color = 'red')
   plt.scatter(points_x, points_y, color = 'red', marker = 'x')
   plt.show()

if __name__ == '__main__':
   traj_x, traj_y = conjugate_gradient(func4, func4_grad_x, func4_grad_y)
   # print(traj_x)
   # print(line_search(func4, 4, 4, func4_grad_x(4, 4), func4_grad_y(4, 4)))
   # print(traj_x)
   draw_plot(func4, traj_x, traj_y)
   print(traj_x[-1], traj_y[-1])