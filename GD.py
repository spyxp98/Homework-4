# TODO: Сделать градиентный спуск с динамичиским шагом

import math

func4 = lambda x, y: x**2 + y**2
func4_grad_x = lambda x, y: 2*x
func4_grad_y = lambda x, y: 2*y
dist = lambda x1, y1, x2, y2: math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def line_search(func, x_start, y_start, dir_x, dir_y):
   # dir = grad_y(x_start, y_start) / grad_x(x_start, y_start)
   step = 0.01
   x = x_start - step * dir_x
   y = y_start - step * dir_y
   while func(x, y) < func(x_start, y_start):
      x -= step * dir_x
      y -= step * dir_y

   return dist(x, y, x_start, y_start)

if __name__ == '__main__':
   print(line_search(func4, -4, 4, -8, 8))