from graphics import *
import numpy as np


win = GraphWin('Graph', 640, 480)
win.setBackground("white")

LTMargin = 100
TPMargin = 100

xmax = 640 - LTMargin 
ymax = 480 - TPMargin 



def rect(cpu,size):

	rectangle = Rectangle(Point(LTMargin, 480 - TPMargin - np.int(cpu/10) ),Point((size/50000) + LTMargin, 480 - TPMargin ))
	rectangle.draw(win)

	rectangle2 = Rectangle(Point(LTMargin, TPMargin-50 ),Point(LTMargin+15, TPMargin - 45))
	rectangle2.setFill('black')
	rectangle2.draw(win)
	rec2_message = Text(Point(LTMargin+40, TPMargin - 48), '--> CPU')
	rec2_message.setSize(8)
	rec2_message.draw(win)

	rectangle3 = Rectangle(Point(LTMargin+100, TPMargin-50 ),Point(LTMargin+115, TPMargin - 45))
	rectangle3.setFill('red')
	rectangle3.draw(win)
	rec3_message = Text(Point(LTMargin+175, TPMargin - 48), '--> GPU1 or Row Wise')
	rec3_message.setSize(8)
	rec3_message.draw(win)

	rectangle4 = Rectangle(Point(LTMargin+250, TPMargin-50), Point(LTMargin+265, TPMargin-45))
	rectangle4.setFill('blue')
	rectangle4.draw(win)
	rec4_message = Text(Point(LTMargin+325, TPMargin - 48), '--> GPU2 or Grid Wise')
	rec4_message.setSize(8)
	rec4_message.draw(win)


def message():

	message = Text(Point(win.getWidth()/2, 20), 'CPU vs GPU execution time')
	message.draw(win)


	x_message1 = Text(Point(win.getWidth()/2, win.getHeight()-TPMargin + 35), '______________ ')
	x_message1.setSize(12)
	x_message1.draw(win)

	x_message2 = Text(Point(win.getWidth()/2 + 60, win.getHeight()-TPMargin + 41), '> ')
	x_message2.setSize(12)
	x_message2.draw(win)

	x_message3 = Text(Point(win.getWidth()/2, win.getHeight()-TPMargin + 55), 'No. of Data')
	x_message3.setSize(8)
	x_message3.setStyle('bold')
	x_message3.draw(win)

	x_message4 = Text(Point(win.getWidth()/2, win.getHeight()-TPMargin + 67), '(in 50K)')
	x_message4.setSize(8)
	x_message4.setStyle('italic')
	x_message4.draw(win)


	y_message1 = Text(Point(win.getHeight()/2-180, win.getWidth()/2-100), '^')
	y_message1.setSize(14)
	y_message1.draw(win)

	y_message2 = Text(Point(win.getHeight()/2-180, win.getWidth()/2-98), '|')
	y_message2.setSize(12)
	y_message2.draw(win)

	y_message3 = Text(Point(win.getHeight()/2-180, win.getWidth()/2-82), '|')
	y_message3.setSize(12)
	y_message3.draw(win)

	y_message4 = Text(Point(win.getHeight()/2-180, win.getWidth()/2-66), '|')
	y_message4.setSize(12)
	y_message4.draw(win)

	y_message5 = Text(Point(win.getHeight()/2-210, win.getWidth()/2-82), 'Time')
	y_message5.setSize(8)
	y_message5.setStyle('bold')
	y_message5.draw(win)

	y_message6 = Text(Point(win.getHeight()/2-210, win.getWidth()/2-72), '(in Sec.)')
	y_message6.setSize(8)
	y_message6.setStyle('italic')
	y_message6.draw(win)
	win.getMouse()
	win.close()


def outGrid(size, cpu):
	x_grid_1 = Text(Point(LTMargin + np.int(size/50000), win.getHeight()-TPMargin + 14), np.int(size/50000))
	x_grid_1.setSize(7)
	x_grid_1.draw(win)
	x_grid_1_1 = Text(Point(LTMargin + np.int(size/50000), win.getHeight()-TPMargin + 3), '|')
	x_grid_1_1.setSize(7)
	x_grid_1_1.draw(win)

	y_grid_1 = Text(Point(LTMargin - 17, win.getHeight()-TPMargin - np.int(cpu/10)), np.int(cpu/10))
	y_grid_1.setSize(7)
	y_grid_1.draw(win)
	y_grid_1_1 = Text(Point(LTMargin - 3, win.getHeight()-TPMargin - np.int(cpu/10)), '-')
	y_grid_1_1.setSize(7)
	y_grid_1_1.draw(win)


def lineDraw(x_axis, y_axis, color):
	xpos = 0
	ypos = 0
	xold = LTMargin
	yold = ymax 
	for i in xrange(0, len(x_axis)):
		xpos = LTMargin + x_axis[i]/50000
		ypos = ymax - (y_axis[i]/10)
		line = Line( Point(xpos,ypos), Point(xold,yold))
		line.setFill(color)
		line.draw(win)
		xold = xpos
		yold = ypos