# TODO: реализовать метод градиентного спуска и метод 
#       сопряженных градиентов.
# тестим на функции f(x, y) = cos^2(x)*sin(y)
# grad f = {-2 cos(x)*sin(x)*sin(y), cos^2(x)*cos(y)}
import math
import matplotlib.pyplot as plt
import numpy as np

func2 = lambda x, y: math.cos(math.pi * x) * (y-1)**2
func2_grad_x = lambda x, y: -math.pi * math.sin(math.pi * x) * (y-1)**2
func2_grad_y = lambda x, y: math.cos(math.pi * x) * 2 * (y-1)

func1 = lambda x, y: (math.cos(x))**2 * math.sin(y)
func1_grad_x = lambda x, y: -2*math.cos(x)*math.sin(x)*math.sin(y)
func1_grad_y = lambda x, y: (math.cos(x))**2 * math.cos(y)

dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
def gradient_descent(grad_x, grad_y):
   # массив для хранения траектории
   traj_x = list()
   traj_y = list()
   # параметры алгоритма
   max_iterations = 1000
   gamma = 0.01
   precision = 0.001
   # начальная точка
   x_start, y_start = 3, 1.8
   iter = 0
   last_step_size = 1
   cur_x, cur_y = x_start, y_start
   while iter < max_iterations and precision < last_step_size:
      prev_x, prev_y = cur_x, cur_y
      cur_x -= gamma * grad_x(prev_x, prev_y)
      cur_y -= gamma * grad_y(prev_x, prev_y)
      last_step_size = dist(prev_x, prev_y, cur_x, cur_y)
      traj_x.append(cur_x)
      traj_y.append(cur_y)
      iter += 1
   
   print(f'Minimum at x = {cur_x}, y = {cur_y}. \n Number of iterations - {iter}')
   return traj_x, traj_y

def loose_line_search(func, x_start, y_start, func_grad_x, func_grad_y):
   step = 0.01
   cnt = 1
   dir = func1_grad_y(x_start, y_start)/func1_grad_x(x_start, y_start) 
   x_l, y_l = x_start, y_start
   x, y = x_start + step, y_start + step*dir
   x_r, y_r = x_start + 2 * step, y_start + 2 * step * dir
   while not func(x, y) < func(x_l, y_l) and func(x, y) < func(x_r, y_r):
      x_l += step
      y_l += dir * step 
      x += step
      y += dir * step 
      x_r += step
      y_r += dir * step 
      cnt += 1
   return cnt * step

def conjugate_gradient(func, grad_x, grad_y, type):
   max_iterations = 10
   precision = 0.0001
   x_start, y_start = 1, 1
   traj_x = list()
   traj_y = list()
   iter = 0
   x_prev, y_prev = x_start, y_start
   x_dir_new = grad_x(x_start, y_start)
   y_dir_new = grad_y(x_start, y_start)
   upd_dir = 0
   x_new = x_start
   y_new = y_start
   while math.sqrt(grad_x(x_new, y_new)**2 + grad_y(x_new, y_new)**2) > precision or iter < max_iterations:
      x_prev = x_new
      y_prev = x_new
      x_dir_prev = x_dir_new
      y_dir_prev = y_dir_new
      # if type == 'PR':
         # beta = (x_new * (x_new - x_prev) + y_new * (y_new - y_prev))/(x_prev ** 2 + y_prev ** 2)
      upd_dir = 1
      x_dir_new = grad_x(x_new, y_new) + upd_dir * x_dir_prev
      y_dir_new = grad_y(x_new, y_new) + upd_dir * y_dir_prev
      step = loose_line_search(func, x_new, y_new, grad_x, grad_y)
      x_new = x_prev + step * x_dir_new
      y_new = y_prev + step * y_dir_new
      traj_x.append(x_new)
      traj_y.append(y_new)
      iter += 1

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
   traj_x, traj_y = gradient_descent(func1_grad_x, func1_grad_y)
   draw_plot(func1, traj_x, traj_y)
