#!~/anaconda/envs/Transition/bin/python python
#!/usr/bin/env python
# coding: utf-8


# In[6]:
import argparse

from PIL import Image
import subprocess
# import imageio
from PyPDF2 import PdfFileWriter, PdfFileReader
import io
import subprocess

import numpy as np
import ot
import matplotlib.pyplot as plt
# from scipy.linalg import toeplitz
# from scipy.stats import mode
import time
from scipy.sparse import coo_matrix, csc_matrix
# import math
from sklearn.cluster import KMeans
import pickle


# In[2]:


def normalise(source, target):
    return np.divide(source, np.sum(source))


# In[3]:


def scale_recursive_OT(X1, X2, K, time_init, color_reg, pos_reg, first=False, second=False):
    if first:
        print("shape input", X1.shape, X2.shape)
    if K is None:
        K = 10
    m = len(X1)
    n = len(X2)

    kmeans1 = KMeans(n_clusters=K, random_state=0, n_init=2).fit(X1)
    kmeans2 = KMeans(n_clusters=K, random_state=0, n_init=2).fit(X2)
    if first:
        print("After kmeans fit", time.time() - time_init)
    M = ((kmeans1.cluster_centers_[:, np.newaxis, :] -
          kmeans2.cluster_centers_[np.newaxis, :, :]) ** 2).sum(axis=2) ** 0.5
    kmeans1_predict = kmeans1.predict(X1)
    kmeans2_predict = kmeans2.predict(X2)
    unique1, counts1_ = np.unique(kmeans1_predict, return_counts=True)
    unique2, counts2_ = np.unique(kmeans2_predict, return_counts=True)
    counts1 = np.zeros(K)
    counts2 = np.zeros(K)
    counts1[unique1] = counts1_ / counts1_.sum()
    counts2[unique2] = counts2_ / counts2_.sum()
    T_partial = ot.emd(a=counts1, b=counts2, M=M)
    #     plt.show()
    #     plt.figure(1)
    #     plt.plot(kmeans1.cluster_centers_[:, 1], kmeans1.cluster_centers_[:, 0])
    #     plt.plot(kmeans2.cluster_centers_[:, 1], kmeans2.cluster_centers_[:, 0])
    #     print(T_partial, counts1, counts2, M)
    #     plt.show()
    #     auie
    if first:
        print("After OT", time.time() - time_init)
    T_row, T_col, T_value = [], [], []
    for i in range(T_partial.shape[0]):
        if first:
            print(".", end="")
        for j in range(T_partial.shape[1]):
            if T_partial[i, j] > 0:
                argwhere1i = np.argwhere(kmeans1_predict == i).squeeze()
                argwhere2j = np.argwhere(kmeans2_predict == j).squeeze()
                X1i = X1[argwhere1i]
                X2j = X2[argwhere2j]
                if len(X1i.shape) != 2 or len(X2j.shape) != 2:
                    continue
                if X1i.shape[0] <= K or X2j.shape[0] <= K:
                    M = ((X1i[:, np.newaxis, :] - X2j[np.newaxis, :, :]) ** 2).sum(axis=2) ** 0.5
                    T_ = ot.emd(a=ot.unif(X1i.shape[0]),
                                b=ot.unif(X2j.shape[0]),
                                M=M) * T_partial[i, j]
                    T_coo = coo_matrix(T_)
                    T = [T_coo.row, T_coo.col, T_coo.data]
                else:
                    if first:
                        second = True
                    else:
                        second = False
                    T = scale_recursive_OT(X1i, X2j, K, time_init, color_reg, pos_reg, second=second)
                    #                     print(type(T[2]))
                    #                     print(T[2], counts1[i], counts2[j])
                    T[2] = T[2] * T_partial[i, j]
                for k in range(len(T[0])):
                    T_row.append(argwhere1i[T[0][k]])
                    T_col.append(argwhere2j[T[1][k]])
                    T_value.append(T[2][k])
    return [T_row, T_col, np.array(T_value) / np.sum(T_value)]


#     return np.concatenate((np.array(T_row)[np.newaxis],
#                            np.array(T_col)[np.newaxis],
#                            np.array(T_value)[np.newaxis]), axis=0)


# In[4]:


def pdf_page_to_png(src_pdf, pagenum=0, resolution=72, ):
    """
    Returns specified PDF page as wand.image.Image png.
    :param PyPDF2.PdfFileReader src_pdf: PDF from which to take pages.
    :param int pagenum: Page number to take.
    :param int resolution: Resolution for resulting png in DPI.
    """
    dst_pdf = PyPDF2.PdfFileWriter()
    dst_pdf.addPage(src_pdf.getPage(pagenum))

    pdf_bytes = io.BytesIO()
    dst_pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    img = Image(file=pdf_bytes, resolution=resolution)
    img.convert("png")

    return img


def preprocess(path, path_slide="./Presentation_OT.pdf", quality=(250, 250), quality_pres=(500, 500)):
    inputpdf = PdfFileReader(open(path_slide, "rb"))
    for i in range(inputpdf.numPages):
        output = PdfFileWriter()
        output.addPage(inputpdf.getPage(i))
        with open(path + "./pdf_generated/" + str(i) + ".pdf", "wb") as outputStream:
            output.write(outputStream)

        output_bash = subprocess.check_output(['pdftoppm',
                                               path + "./pdf_generated/" + str(i) + ".pdf",
                                               path + "./png_generated/" + str(i),
                                               "-png",
                                               "-rx",
                                               str(quality[0]),
                                               "-ry",
                                               str(quality[1])])
        output_bash = subprocess.check_output(['pdftoppm',
                                               path + "./pdf_generated/" + str(i) + ".pdf",
                                               path + "./png_generated/" + str(i) + "_pres",
                                               "-png",
                                               "-rx",
                                               str(quality_pres[0]),
                                               "-ry",
                                               str(quality_pres[1])])
    return i + 1


def main(path="./pdftoimage/",
         path_slide="./Presentation_OT.pdf",
         K=10,
         color_reg=1,
         pos_reg=1,
         pos=None,
         quality=(250, 250),
         quality_pres=(500, 500)):
    number_slide = preprocess(path_slide=path_slide, path=path, quality=quality, quality_pres=quality_pres)

    number_color = 3

    time_init = time.time()
    I_PIL = []
    I = []
    X = [0] * number_slide
    #     I_color = [[0] * number_color for _ in range(number_slide)]
    #     I_color[0][0] = 1
    #     I_color_sum = np.zeros((I, 3))
    max_shape_X = 0
    for i in range(number_slide):
        I_PIL.append(Image.open(path + "png_generated/" + str(i) + "-1.png"))
        if pos is None:
            I_PIL[i] = 1 - (np.array(I_PIL[i])) / 255
        else:
            I_PIL[i] = 1 - (np.array(I_PIL[i])[pos[0]:max(pos[1], I_PIL[i].size[0]),
                            pos[2]:max(pos[3], I_PIL[i].size[1]),
                            :3]) / 255
        X_pos = list(np.where(np.sum(I_PIL[i], axis=2) != 0))

        X[i] = np.concatenate((X_pos[0][:, np.newaxis],
                               X_pos[1][:, np.newaxis],
                               I_PIL[i][X_pos[0], X_pos[1], :]),
                              axis=1)

        max_shape_X = np.maximum(X[i].shape[0], max_shape_X)

    print("After load", time.time() - time_init)

    print(X[0].shape)

    for i in range(len(X) - 1):
        print("")
        print(i)
        T = scale_recursive_OT(X[i], X[i + 1], K=K, time_init=time_init,
                               color_reg=color_reg,
                               pos_reg=pos_reg,
                               first=True)
        dict_plot = {}
        dict_plot["T"] = T
        dict_plot["X"] = X
        dict_plot["I_PIL"] = [I_PIL[i], I_PIL[i + 1]]
        dict_plot["path"] = path
        with open(path + "pickle_T/" + path_slide[2:-4] + str(i) + ".pickle", 'wb') as handle:
            pickle.dump(dict_plot, handle)


def plot(path="./pdftoimage/",
         path_slide="./Presentation_OT.pdf",
         t_list=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999],
         height=10,
         width=10,
         save=False,
         plot_fig=True):
    i = -1
    while True:
        i = i + 1
        try:
            print(path + "pickle_T/" + path_slide[2:-4] + str(i) + ".pickle")
            with open(path + "pickle_T/" + path_slide[2:-4] + str(i) + ".pickle", 'rb') as handle:
                dict_plot = pickle.load(handle)
        except:
            break
        time_init = time.time()
        T = dict_plot["T"]
        X = dict_plot["X"]
        I_PIL = dict_plot["I_PIL"]
        path_generated = path + "png_transition_generated/"

        T_pos = np.concatenate((np.array(T[0])[np.newaxis], np.array(T[1])[np.newaxis]), axis=0)
        T_val = np.array(T[2])
        #         T_coo = coo_matrix(T)
        #         T = [T_coo.row, T_coo.col, T_coo.data]

        #         u, index = np.unique(T_pos,
        #                              axis=1,
        #                              return_index=True)
        #         print(index.shape)
        #         print(u.shape)
        # #         auie
        # #         T_pos[0] = T_pos[0][index]
        # #         T_pos = u
        #         print(T_pos.shape, T_val.shape)
        # #         T_val = T_val[index]
        #         print(T_pos.shape, T_val.shape)

        #         T_pos[0] = np.clip(T_pos[0], 0, max(I_PIL[i].shape[0], I_PIL[i].shape[0]) - 1)
        #         T_pos[1] = np.clip(T_pos[1], 0, max(I_PIL[i].shape[1], I_PIL[i].shape[1]) - 1)

        #         r = max(I_PIL[i].shape[0], I_PIL[i+1].shape[0]) / max(I_PIL[i].shape[1], I_PIL[i+1].shape[1])
        #         fig.set_figheight(height * (len(t_list) + 2))
        #         fig.set_figwidth(width)
        fig = plt.figure(figsize=(height, width))
        plt.imshow(1 - I_PIL[0])
        plt.axis('off')
        if save:
            plt.savefig(path_generated + str(i) + "_0" + ".png", bbox_inches='tight', pad_inches=0)
        if plot_fig:
            plt.show()
        for j, t in enumerate(reversed(t_list)):
            print("t", t, j, time.time() - time_init)

            I_t = np.zeros((max(I_PIL[0].shape[0], I_PIL[1].shape[0]),
                            max(I_PIL[0].shape[1], I_PIL[1].shape[1]),
                            3))
            #             for k in range(len(T[0])):
            #                 x_pos_k,y_pos_k,color1,color2,color3 = X[i][T_pos[0][k]] * t + X[i + 1][T_pos[1][k]] * (1 - t)
            #                 I_t[int(x_pos_k), int(y_pos_k)] += np.array([color1, color2, color3]) * T_val[k]
            pos = X[i][T_pos[0]] * t + X[i + 1][T_pos[1]] * (1 - t)
            I_t[pos[:, 0].astype(int), pos[:, 1].astype(int)] += pos[:, 2:] * T_val[:, np.newaxis]

            I_t = I_t * ((X[i][:, 2:].sum() * t + X[i + 1][:, 2:].sum() * (1 - t)) / I_t.sum())
            #             print("before clip", I_t.mean(), I_PIL[i].mean(), I_PIL[i + 1].mean())
            I_t = np.clip(I_t, 0, 1)
            #             print("X", X[i][:, 2:].mean(), X[i + 1][:, 2:].mean())
            #             print(I_t.mean(), I_PIL[i].mean(), I_PIL[i + 1].mean())
            fig = plt.figure(figsize=(height, width))
            plt.imshow(1 - I_t)
            plt.axis('off')
            if save:
                plt.savefig(path_generated + str(i) + "_" + str(t) + ".png", bbox_inches='tight', pad_inches=0)
            if plot_fig:
                plt.show()
        fig = plt.figure(figsize=(height, width))
        plt.imshow(1 - I_PIL[1])
        plt.axis('off')
        if save:
            plt.savefig(path_generated + str(i) + "_1" + ".png", bbox_inches='tight', pad_inches=0)
        if plot_fig:
            plt.show()
        plt.close('all')
    dict_plot2 = {}
    dict_plot2["t_list"] = t_list
    dict_plot2["I"] = i
    print(dict_plot2)
    with open(path + "pickle_T/" + path_slide[2:-4] + ".pickle", 'wb') as handle:
        pickle.dump(dict_plot2, handle)


# In[100]:


def create_gif(path="./pdftoimage/",
               path_slide="./Presentation_OT.pdf", duration=200):
    # def make_gif(path_gif, input_png_list):
    #     clips = [mpy.ImageClip(i).set_duration(0.1)
    #              for i in input_png_list]
    #     concat_clip = mpy.concatenate_videoclips(clips, method="compose")
    #     concat_clip.write_gif(path_gif, fps=2)

    with open(path + "pickle_T/" + path_slide[2:-4] + ".pickle", 'rb') as handle:
        dict_plot = pickle.load(handle)

    t_list = dict_plot["t_list"]
    I = dict_plot["I"]
    t_list = [1] + t_list + [0]
    print(I)
    for i in range(I):
        print(i)
        images = []
        #         images.append(imageio.imread(path + "png_generated/" + str(i) + "-" + str(1) + ".png"))
        images.append(Image.open(path + "png_generated/" + str(i) + "-" + str(1) + ".png"))
        for t in reversed(t_list):
            images.append(Image.open(path + "png_transition_generated/" + str(i) + "_" + str(t) + ".png"))
            images[-1] = images[-1].resize((images[0].size), Image.ANTIALIAS)
        images.append(Image.open(path + "png_generated/" + str(i + 1) + "-" + str(1) + ".png"))
        images[0].save(path + "gif_generated/" + str(i) + "-" + str(i + 1) + '.gif',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration)
        images.reverse()
        images[0].save(path + "gif_generated/" + str(i + 1) + "-" + str(i) + '.gif',
                       save_all=True,
                       append_images=images[1:],
                       duration=duration)
    return I


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GROMAP')
    parser.add_argument('--qualityx', type=int, default=50)
    parser.add_argument('--qualityy', type=int, default=50)
    parser.add_argument('--qualityx_pres', type=int, default=500)
    parser.add_argument('--qualityy_pres', type=int, default=500)
    parser.add_argument('--duration', type=int, default=300)
    parser.add_argument('--number_slide', type=int, default=None)
    parser.add_argument('--K', type=int, default=100)
    parser.add_argument('-T', '--T', action="store_false")
    parser.add_argument('-P', '--P', action="store_false")
    parser.add_argument('-G', '--G', action="store_false")
    parser.add_argument('-V', '--V', action="store_false")
    # parser.add_argument('--t_list', type=str, default="0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999")
    parser.add_argument('--t_list', type=str, default="0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.995,0.999")
    args = parser.parse_args()
    I = args.number_slide
    t_list = (args.t_list).split(",")
    for i in range(len(t_list)):
        t_list[i] = float(t_list[i])
    if args.T:
        main(K=args.K, quality=(args.qualityx, args.qualityy), quality_pres=(args.qualityx_pres, args.qualityy_pres))
    if args.P:
        plot(save=True, plot_fig=False, t_list=t_list)
    if args.G:
        I = create_gif(duration=args.duration)
    assert I is not None
    if args.V:
        subprocess.check_call(["./video.sh", str(I - 1)])

    # subprocess.call("webm -i ./ pdftoimage / gif_generated /" + str(i) + "-" + str(j) +".gif. / pdftoimage / video_generated /" +
    #                 str(i) + "-" + str(j) + ".mp4",
    #                 shell=True)
    # print(("webm -i ./pdftoimage/gif_generated/" + str(i) + "-" + str(j) +".gif ./pdftoimage/video_generated/" +
    #        str(i) + "-" + str(j) + ".mp4").split())
    # a = subprocess.run(("webm -i ./pdftoimage/gif_generated/" + str(i) + "-" + str(j) +".gif ./pdftoimage /video_generated /" +
    #                 str(i) + "-" + str(j) + ".mp4").split(), stdout=subprocess.PIPE, shell=True)
# for i in range(2,20):
#    print("            <div>")
#    print('                <video src="./pdftoimage/video_generated/' + str(i) + '-' + str(i+1) +'.mp4"></video>')
#    print('                <video src="./pdftoimage/video_generated/' + str(i+1) + '-' + str(i) +'.mp4"></video>')
#    print("            </div>")
#    print('            <img src="./pdftoimage/png_generated/' + str(i+1) + '-1.png"/>')
