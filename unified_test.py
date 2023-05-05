from time import perf_counter as clock
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
import config
from gym.wrappers.monitor import Monitor

from numpy import linspace,zeros_like,ones, ones_like
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import randint


def Chessboard(size, mask, bx):
    x = linspace(0, size, 1000)
    vertical = zeros_like(x)
    points = []
    bx.axis('off')
    for item in range(len(mask)):
        if mask[item] == 1:
            points.append([item//size, item%size])
    for point in points:
        bx.scatter(point[0], point[1],color = 'green',s = 70,zorder = 2)
        bx.scatter(point[0], point[1],color = 'white',s = 10,zorder = 3)

    bx.plot(vertical + 0, x, color="gray", linestyle='-', zorder=1)
    bx.plot(x, vertical + 0, color="gray", linestyle='-', zorder=1)

    for i in range(1,size+1):
        bx.plot(vertical+i, x, color="gray",linestyle = '--', zorder = 1)
        bx.plot(x, vertical+i, color="gray",linestyle = '--', zorder = 1)

    bx.quiver(10, 0, 12, 0, linestyle='-', linewidth=1.5, color='gray')
    bx.quiver(0, 10, 0,12, linestyle='-', linewidth=1.5, color='gray',)
    bx.text(10,0-0.8,'x',fontsize = 20,color = 'gray')
    bx.text(0-0.8,10,'y',fontsize = 20,color = 'gray')
    for i in range(10):
        bx.text(i,0-0.7,'{}'.format(i),color = 'gray',fontsize = 15)
        bx.text(0-0.7,i,'{}'.format(i),color = 'gray',fontsize = 15)


def Discrete(ax, size, mask):
    x = linspace(0, size, 1000)
    vertical = zeros_like(x)
    points = []
    for item in range(len(mask)):
        if mask[item] == 1:
            points.append([item//size, item%size, size])
    for point in points:
        ax.scatter(point[0], point[1],point[2], s = 5, color = 'red',zorder = 2)
    for i in range(size+1):
        ax.plot3D(vertical+i, x, ones_like(x) * size, linewidth= '0.5', linestyle='--', color="gray",zorder = 1)
        ax.plot3D(x, vertical+i, ones_like(x) * size, linewidth= '0.5', linestyle='--', color="gray",zorder = 1)

def choose_color():
    color = ['gold', 'springgreen','pink','aquamarine','cyan']
    index = randint(0,len(color)-1)
    return color[index]

def plot_container(game, ax):
    game.container.color = 'gray'
    game.container.plot_linear_cube(ax,"",linestyle = '--')


    size = game.bin_size[0]
    ax.quiver(0, 0, 0, size * 1.2, 0, 0, linestyle='-', linewidth=1.5, color='r', arrow_length_ratio=.05)
    ax.quiver(0, 0, 0, 0, size * 1.2, 0, linestyle='-', linewidth=1.5, color='g', arrow_length_ratio=.08)
    ax.quiver(0, 0, 0, 0, 0, size * 1.2, linestyle='-', linewidth=1.5, color='b', arrow_length_ratio=.08)
    ax.text3D(size * 1.3, 0, -1, 'x', fontsize=15, color='r')
    ax.text3D(0, size * 1.3, 0 + 0.5, 'y', fontsize=15, color='g')
    ax.text3D(0 - 0.5, 0 - 0.5, size * 1.3, 'z', fontsize=15, color='b')
    # infor_around_container(game.container,ax)

def draw_3D_box(env,container = [10,10,10], ax=None, info=None):
    ax.grid(False)
    ax.axis('off')
    plot_container(env, ax)
    boxes = env.space.boxes
    tray_box = env.space.tray_box
    
    if not (len(boxes) == 0):
        for i in range(len(boxes)-1):
            boxes[i].set_color(choose_color())
            boxes[i].plot_opaque_cube(ax, str(i), alpha=1)
        boxes[len(boxes) - 1].set_color(choose_color())
        boxes[len(boxes) - 1].plot_opaque_cube(ax, str(len(boxes) - 1), alpha=0.5)
        boxes[len(boxes) - 1].plot_opaque_cube(ax, str(len(boxes) - 1), alpha=1.0, demo=True)
    
    if tray_box is not None:
        tray_box.set_color('green')
        tray_box.plot_opaque_cube(ax, str(len(boxes)+1))

    size = env.bin_size[0]
    # mask = ones(size**2)
    mask = env.get_possible_position()
    mask = mask.reshape(size**2)
    
    Discrete(ax,size,mask)
    
    ax.text(15, 10, 0, "space utilization: " + str(info['ratio']), fontsize=15)

    ax.set_xlim3d(0, container[0] * 1.5)
    ax.set_ylim3d(0, container[0] * 2)
    ax.set_zlim3d(0, container[0] * 1.5)
    ax.set_xlabel("x-label", color='r')
    ax.set_ylabel("y-label", color='g')
    ax.set_zlabel("z-label", color='b')


def run_sequence(nmodel, raw_env, preview_num, c_bound, ax, bx):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = clock()
    size = env.bin_size[0]
    while True:
        box_list = env.box_creator.preview(preview_num)
        # print(box_list)
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])
        
        mask = env.get_possible_position()
        mask = mask.reshape(size ** 2)
        plt.figure(1)
        plt.cla()
        plt.figure(2)
        plt.cla()
        Chessboard(size, mask, bx)
        draw_3D_box(env, env.bin_size, ax, info)
        ax.figure.canvas.draw_idle()
        bx.figure.canvas.draw_idle()
        plt.draw()
        plt.pause(1)
        
        if done:
            end = clock()
            print('Time cost:', end-start)
            print('Ratio:', info['ratio'])
            return info['ratio'], info['counter'], end-start,default_counter/box_counter
        box_counter += 1
        default_counter += int(default)

def unified_test(url, config):
    fig1 = plt.figure(1)
    fig2 = plt.figure(2,figsize = (5,5))
    ax = fig1.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.axis('off')
    bx = fig2.add_subplot(111)

    nmodel = nnModel(url, config)
    data_url = config.data_dir+config.data_name
    env = gym.make(config.env_name, _adjust_ratio=0, adjust=False,
                    box_set=config.box_size_set,
                    container_size=config.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=config.enable_rotation,
                    data_type=config.data_type)
    print('Env name: ', config.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', config.cases)
    print('pruning threshold: ', config.pruning_threshold)
    print('Known item number: ', config.preview)
    # draw_3D_box(env, env.bin_size, ax)
    times = config.cases
    ratios = []
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = config.pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate = run_sequence(nmodel, env, config.preview, c_bound, ax, bx)
        avg_ratio += ratio
        ratios.append(ratio)
        avg_counter += counter
        avg_time += time
        avg_drate += depen_rate

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(avg_ratio/times))
    print('average put item number: %.4f'%(avg_counter/times))
    print('average sequence time: %.4f'%(avg_time/times))
    print('average time per item: %.4f'%(avg_time/avg_counter))
    print('----------------------------------------------')

if __name__ == '__main__':
    config.cases = 100
    config.preview = 1
    unified_test('pretrained_models/default_cut_2.pt', config)
    # config.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', config)

    