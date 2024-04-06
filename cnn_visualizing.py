from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
import matplotlib.pyplot as plt
from keras.utils import plot_model
import visualkeras

# Create a Sequential model
model = Sequential()
model.add(Input(shape=(150,150,3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

plot_model(model=model, to_file='cnn.png')

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcdefaults()
# from matplotlib.lines import Line2D
# from matplotlib.patches import Rectangle
# from matplotlib.patches import Circle

# NumDots = 4
# NumConvMax = 8
# NumFcMax = 20
# White = 1.
# Light = 0.7
# Medium = 0.5
# Dark = 0.3
# Darker = 0.15
# Black = 0.

# def add_layer(patches, colors, size=(24, 24), num=5,
#               top_left=[0, 0],
#               loc_diff=[3, -3],
#               ):
#     # add a rectangle
#     top_left = np.array(top_left)
#     loc_diff = np.array(loc_diff)
#     loc_start = top_left - np.array([0, size[0]])
#     for ind in range(num):
#         patches.append(Rectangle(loc_start + ind * loc_diff, size[1], size[0]))
#         if ind % 2:
#             colors.append(Medium)
#         else:
#             colors.append(Light)

# def add_layer_with_omission(patches, colors, size=(24, 24),
#                             num=5, num_max=8,
#                             num_dots=4,
#                             top_left=[0, 0],
#                             loc_diff=[3, -3],
#                             ):
#     # add a rectangle
#     top_left = np.array(top_left)
#     loc_diff = np.array(loc_diff)
#     loc_start = top_left - np.array([0, size[0]])
#     this_num = min(num, num_max)
#     start_omit = (this_num - num_dots) // 2
#     end_omit = this_num - start_omit
#     start_omit -= 1
#     for ind in range(this_num):
#         if (num > num_max) and (start_omit < ind < end_omit):
#             omit = True
#         else:
#             omit = False

#         if omit:
#             patches.append(
#                 Circle(loc_start + ind * loc_diff + np.array(size) / 2, 0.5))
#         else:
#             patches.append(Rectangle(loc_start + ind * loc_diff,
#                                      size[1], size[0]))

#         if omit:
#             colors.append(Black)
#         elif ind % 2:
#             colors.append(Medium)
#         else:
#             colors.append(Light)

# def add_mapping(patches, colors, start_ratio, end_ratio, patch_size, ind_bgn,
#                 top_left_list, loc_diff_list, num_show_list, size_list):

#     start_loc = top_left_list[ind_bgn] \
#         + (num_show_list[ind_bgn] - 1) * np.array(loc_diff_list[ind_bgn]) \
#         + np.array([start_ratio[0] * (size_list[ind_bgn][1] - patch_size[1]),
#                     - start_ratio[1] * (size_list[ind_bgn][0] - patch_size[0])]
#                    )

#     end_loc = top_left_list[ind_bgn + 1] \
#         + (num_show_list[ind_bgn + 1] - 1) * np.array(
#             loc_diff_list[ind_bgn + 1]) \
#         + np.array([end_ratio[0] * size_list[ind_bgn + 1][1],
#                     - end_ratio[1] * size_list[ind_bgn + 1][0]])

#     patches.append(Rectangle(start_loc, patch_size[1], -patch_size[0]))
#     colors.append(Dark)
#     patches.append(Line2D([start_loc[0], end_loc[0]],
#                           [start_loc[1], end_loc[1]]))
#     colors.append(Darker)
#     patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
#                           [start_loc[1], end_loc[1]]))
#     colors.append(Darker)
#     patches.append(Line2D([start_loc[0], end_loc[0]],
#                           [start_loc[1] - patch_size[0], end_loc[1]]))
#     colors.append(Darker)
#     patches.append(Line2D([start_loc[0] + patch_size[1], end_loc[0]],
#                           [start_loc[1] - patch_size[0], end_loc[1]]))
#     colors.append(Darker)

# def label(xy, text, xy_off=[0, 4]):
#     plt.text(xy[0] + xy_off[0], xy[1] + xy_off[1], text,
#              family='sans-serif', size=8)

# # Create the CNN model
# patches = []
# colors = []

# fig, ax = plt.subplots()

# layer_width = 40
# fc_unit_size = 2

# size_list = [(150, 150), (148, 148), (146, 146), (72, 72), (70, 70), (34, 34), (32, 32), (30, 30), (14, 14), (12, 12), (10, 10), (4, 4)]
# num_list = [3, 32, 64, 64, 64, 128, 128, 128, 128, 256, 512, 512, 4]
# x_diff_list = [0, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width, layer_width]
# text_list = ['Inputs'] + ['Feature\nmaps'] * (len(size_list) - 1)
# loc_diff_list = [[3, -3]] * len(size_list)

# num_show_list = list(map(min, num_list, [NumConvMax] * len(num_list)))
# top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]

# for ind in range(len(size_list)-1,-1,-1):
#     add_layer_with_omission(patches, colors, size=size_list[ind],
#                             num=num_list[ind],
#                             num_max=NumConvMax,
#                             num_dots=NumDots,
#                             top_left=top_left_list[ind],
#                             loc_diff=loc_diff_list[ind])
#     label(top_left_list[ind], text_list[ind] + '\n{}@{}x{}'.format(
#         num_list[ind], size_list[ind][0], size_list[ind][1]))

# start_ratio_list = [[0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]
# end_ratio_list = [[0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5], [0.4, 0.5]]
# patch_size_list = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
# ind_bgn_list = range(len(patch_size_list))
# text_list = ['Convolution'] * len(patch_size_list)

# for ind in range(len(patch_size_list)):
#     add_mapping(
#         patches, colors, start_ratio_list[ind], end_ratio_list[ind],
#         patch_size_list[ind], ind,
#         top_left_list, loc_diff_list, num_show_list, size_list)
#     label(top_left_list[ind], text_list[ind] + '\n{}x{} kernel'.format(
#         patch_size_list[ind][0], patch_size_list[ind][1]), xy_off=[26, -65])

# size_list = [(fc_unit_size, fc_unit_size)] * 3
# num_list = [512, 512, 4]
# num_show_list = list(map(min, num_list, [NumFcMax] * len(num_list)))
# x_diff_list = [sum(x_diff_list) + layer_width, layer_width, layer_width]
# top_left_list = np.c_[np.cumsum(x_diff_list), np.zeros(len(x_diff_list))]
# loc_diff_list = [[fc_unit_size, -fc_unit_size]] * len(top_left_list)
# text_list = ['Hidden\nunits'] * (len(size_list) - 1) + ['Outputs']

# for ind in range(len(size_list)):
#     add_layer_with_omission(patches, colors, size=size_list[ind],
#                             num=num_list[ind],
#                             num_max=NumFcMax,
#                             num_dots=NumDots,
#                             top_left=top_left_list[ind],
#                             loc_diff=loc_diff_list[ind])
#     label(top_left_list[ind], text_list[ind] + '\n{}'.format(
#         num_list[ind]))

# text_list = ['Flatten\n', 'Fully\nconnected', 'Fully\nconnected']

# for ind in range(len(size_list)):
#     label(top_left_list[ind], text_list[ind], xy_off=[-10, -65])

# for patch, color in zip(patches, colors):
#     patch.set_color(color * np.ones(3))
#     if isinstance(patch, Line2D):
#         ax.add_line(patch)
#     else:
#         patch.set_edgecolor(Black * np.ones(3))
#         ax.add_patch(patch)

# plt.tight_layout()
# plt.axis('equal')
# plt.axis('off')
# plt.show()
# fig.set_size_inches(12, 6)
# plt.imsave('cnn.jpg')