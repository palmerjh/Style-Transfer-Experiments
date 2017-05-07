# -*- coding: utf-8 -*-

# NOTE: borrowed heavily from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html

"""
Neural Transfer with PyTorch
============================
**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_

Introduction
------------

Welcome! This tutorial explains how to impletment the
`Neural-Style <https://arxiv.org/abs/1508.06576>`__ algorithm developed
by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.

Neural what?
~~~~~~~~~~~~

The Neural-Style, or Neural-Transfer, is an algorithm that taks as
input a content-image (e.g. a tortle), a style-image (e.g. artistic
waves) and return the content of the content-image as if it was
'painted' using the artistic style of the style-image:

.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1

How does it work?
~~~~~~~~~~~~~~~~~

The principe is simple: we define two distances, one for the content
(:math:`D_C`) and one for the style (:math:`D_S`). :math:`D_C` measues
how different is the content between two images, while :math:`D_S`
measures how different is the style between two images. Then, we take a
third image, the input, (e.g. a with noise), and we transform it in
order to both minimize its content-distance with the content-image and
its style-distance with the style-image.
"""

from __future__ import print_function

import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
# import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

content_dir = 'content'
styles_dir = 'styles'

results_dir = 'results'

square_styles = ['field',
                 'clouds',
                 'mountain',
                 'stars_square',
                 'stars2_square',
                 'stars3_square',
                 'aurora_square',
                 'aurora2_square',
                 'aurora3_square',
                 'aurora4_square',
                 'trippy',
                 'trippy2',
                 'trippy3',
                 'trippy4',
                 'tree_square',
                 'cloth_square',
                 'bourbon_square']

content_style_dict = { 'jerbear'    : square_styles,
                       'vegan_pug'    : square_styles,
                       'julianna_dog'    : square_styles,
                       'goat_square'    : square_styles,
                       'gothic_square'    : square_styles,
                       'gorgeous_square'    : square_styles,
                       'wwoof_square'    : square_styles,
                       'plantation'    : square_styles
}

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

######################################################################
# Cuda
# ~~~~
#
# If you have a GPU on your computer, it is preferable to run the
# algorithm on it, especially if you want to try larger networks (like
# VGG). For this, we have ``torch.cuda.is_available()`` that returns
# ``True`` if you computer has an available GPU. Then, we can use method
# ``.cuda()`` that moves allocated proccesses associated with a module
# from the CPU to the GPU. When we want to move back this module to the
# CPU (e.g. to use numpy), we use the ``.cpu()`` method. Finally,
# ``.type(dtype)`` will be use to convert a ``torch.FloatTensor`` into
# ``torch.cuda.FloatTensor`` to feed GPU processes.
#

use_cuda = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


######################################################################
# Load images
# ~~~~~~~~~~~
#
# In order to simplify the implementation, let's start by importing a
# style and a content image of the same dimentions. We then scale them to
# the desired output image size (128 or 512 in the example, depending on gpu
# availablity) and transform them into torch tensors, ready to feed
# a neural network:
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg </_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg </_static/img/neural-style/dancing.jpg>`__. Download these
#     two images and add them to a directory with name ``images``


# desired size of the output image
imsize = 512 if use_cuda else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Scale(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    image = Variable(loader(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


######################################################################
# Imported PIL images has values between 0 and 255. Transformed into torch
# tensors, their values are between 0 and 1. This is an important detail:
# neural networks from torch library are trained with 0-1 tensor image. If
# you try to feed the networks with 0-255 tensor images the activated
# feature maps will have no sense. This is not the case with pre-trained
# networks from the Caffe library: they are trained with 0-255 tensor
# images.
#
# Display images
# ~~~~~~~~~~~~~~
#
# We will use ``plt.imshow`` to display images. So we need to first
# reconvert them into PIL images:
#

unloader = transforms.ToPILImage()  # reconvert into PIL image

# plt.ion()

def imshow(tensor, title=None):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    # image = image.view(3, imsize, imsize)  # remove the fake batch dimension
    image = unloader(image)
    # plt.imshow(image)
    if title is not None:
        pass
        # plt.title(title)
    # plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, fname):
    image = tensor.clone().cpu()  # we clone the tensor to not do changes on it
    # size = image.size()
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)

    image.save(fname)

######################################################################
# Content loss
# ~~~~~~~~~~~~
#
# The content loss is a function that takes as input the feature maps
# :math:`F_{XL}` at a layer :math:`L` in a network fed by :math:`X` and
# return the weigthed content distance :math:`w_{CL}.D_C^L(X,C)` between
# this image and the content image. Hence, the weight :math:`w_{CL}` and
# the target content :math:`F_{CL}` are parameters of the function. We
# implement this function as a torch module with a constructor that takes
# these parameters as input. The distance :math:`\|F_{XL} - F_{YL}\|^2` is
# the Mean Square Error between the two sets of feature maps, that can be
# computed using a criterion ``nn.MSELoss`` stated as a third parameter.
#
# We will add our content losses at each desired layer as additive modules
# of the neural network. That way, each time we will feed the network with
# an input image :math:`X`, all the content losses will be computed at the
# desired layers and, thanks to autograd, all the gradients will be
# computed. For that, we just need to make the ``forward`` method of our
# module returning the input: the module becomes a ''transparent layer''
# of the neural network. The computed loss is saved as a parameter of the
# module.
#
# Finally, we define a fake ``backward`` method, that just call the
# backward method of ``nn.MSELoss`` in order to reconstruct the gradient.
# This method returns the computed loss: this will be usefull when running
# the gradien descent in order to display the evolution of style and
# content losses.
#

class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        self.target = target.detach() * weight
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss


######################################################################
# .. Note::
#    **Important detail**: this module, although it is named ``ContentLoss``,
#    is not a true PyTorch Loss function. If you want to define your content
#    loss as a PyTorch Loss, you have to create a PyTorch autograd Function
#    and to recompute/implement the gradient by the hand in the ``backward``
#    method.
#
# Style loss
# ~~~~~~~~~~
#
# For the style loss, we need first to define a module that compute the
# gram produce :math:`G_{XL}` given the feature maps :math:`F_{XL}` of the
# neural network fed by :math:`X`, at layer :math:`L`. Let
# :math:`\hat{F}_{XL}` be the re-shaped version of :math:`F_{XL}` into a
# :math:`K`\ x\ :math:`N` matrix, where :math:`K` is the number of feature
# maps at layer :math:`L` and :math:`N` the lenght of any vectorized
# feature map :math:`F_{XL}^k`. The :math:`k^{th}` line of
# :math:`\hat{F}_{XL}` is :math:`F_{XL}^k`. We let you check that
# :math:`\hat{F}_{XL} \cdot \hat{F}_{XL}^T = G_{XL}`. Given that, it
# becomes easy to implement our module:
#

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)


######################################################################
# The longer is the feature maps dimension :math:`N`, the bigger are the
# values of the gram matrix. Therefore, if we don't normalize by :math:`N`,
# the loss computed at the first layers (before pooling layers) will have
# much more importance during the gradient descent. We dont want that,
# since the most interesting style features are in the deepest layers!
#
# Then, the style loss module is implemented exactly the same way than the
# content loss module, but we have to add the ``gramMatrix`` as a
# parameter:
#

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.G = self.gram(input)
        self.G.mul_(self.weight)
        self.loss = self.criterion(self.G, self.target)
        return self.output

    def backward(self, retain_variables=True):
        self.loss.backward(retain_variables=retain_variables)
        return self.loss

def get_style_model_and_losses(cnn, style_img, content_img,
                               style_weight=1000, content_weight=1,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential()  # the new Sequential module network
    gram = GramMatrix()  # we need a gram module in order to compute style targets

    # move these modules to the GPU if possible:
    if use_cuda:
        model = model.cuda()
        gram = gram.cuda()

    i = 1
    for layer in list(cnn):
        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)

            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)  # ***

    return model, style_losses, content_losses


######################################################################
# .. Note::
#    In the paper they recommend to change max pooling layers into
#    average pooling. With AlexNet, that is a small network compared to VGG19
#    used in the paper, we are not going to see any difference of quality in
#    the result. However, you can use these lines instead if you want to do
#    this substitution:
#
#    ::
#
#        # avgpool = nn.AvgPool2d(kernel_size=layer.kernel_size,
#        #                         stride=layer.stride, padding = layer.padding)
#        # model.add_module(name,avgpool)



######################################################################
# Gradient descent
# ~~~~~~~~~~~~~~~~
#
# As Leon Gatys, the author of the algorithm, suggested
# `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__,
# we will use L-BFGS algorithm to run our gradient descent. Unlike
# training a network, we want to train the input image in order to
# minimise the content/style losses. We would like to simply create a
# PyTorch  L-BFGS optimizer, passing our image as the variable to optimize.
# But ``optim.LBFGS`` takes as first argument a list of PyTorch
# ``Variable`` that require gradient. Our input image is a ``Variable``
# but is not a leaf of the tree that requires computation of gradients. In
# order to show that this variable requires a gradient, a possibility is
# to construct a ``Parameter`` object from the input image. Then, we just
# give a list containing this ``Parameter`` to the optimizer's
# constructor:
#

def get_input_param_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


######################################################################
# **Last step**: the loop of gradient descent. At each step, we must feed
# the network with the updated input in order to compute the new losses,
# we must run the ``backward`` methods of each loss to dynamically compute
# their gradients and perform the step of gradient descent. The optimizer
# requires as argument a "closure": a function that reevaluates the model
# and returns the loss.
#
# However, there's a small catch. The optimized image may take its values
# between :math:`-\infty` and :math:`+\infty` instead of staying between 0
# and 1. In other words, the image might be well optimized and have absurd
# values. In fact, we must perform an optimization under constraints in
# order to keep having right vaues into our input image. There is a simple
# solution: at each step, to correct the image to maintain its values into
# the 0-1 interval.
#

def run_style_transfer(cnn, content_img, style_img, input_img, outfile, num_steps=600,
                       style_weight=1000, content_weight=1, findMin=True):
    """Run the style transfer."""
    if findMin:
        print('First run over max epochs...')
    else:
        print('Second run over optimum epochs...')

    print(num_steps)

    copies = {'cnn' :   copy.deepcopy(cnn),
              'content_img' : content_img.clone(),
              'style_img' : style_img.clone(),
              'input_img' : input_img.clone() }

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        style_img, content_img, style_weight, content_weight)
    input_param, optimizer = get_input_param_optimizer(input_img)

    # keeps track of what number of epochs yields the minimum score
    min_nEpochs = [0, 10000]
    # cur_nEpochs = [0, 0]

    if findMin:
        out = open(outfile, 'w')

    print('Optimizing..')
    run = [0]
    while run[0] < num_steps:

        def closure():

            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            total_score = style_score + style_score
            #print(total_score)
            total_score_scalar = total_score.data[0]
            if total_score_scalar < min_nEpochs[1]:
                min_nEpochs[0] = run[0]
                min_nEpochs[1] = total_score_scalar
            #
            # # don't ask me why this is required cuz I don't know
            # if run[0] > num_steps:
            #     print('Why am I here?')
            #     return Variable(torch.Tensor([0.0]))
            # elif run[0] == num_steps:
            #     print('should be done now')
            #     #optimizer.step(closure)
            # else:
            #     pass
            #     #optimizer.step(closure)
            #
            # cur_nEpochs[0] = run[0]
            # cur_nEpochs[1] = total_score_scalar
            #
            if run[0] % 10 == 0:
                print('cur: %d\t%f' % (run[0], total_score_scalar))
                print('min: %d\t%f' % (min_nEpochs[0], min_nEpochs[1]))

            if run[0] % 50 == 0:
                out1 = "run {}:".format(run)
                print(out1)
                out2 = 'Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.data[0], content_score.data[0])
                print(out2)
                print()

                if findMin:
                    out.write(out1 + '\n')
                    out.write(out2 + '\n\n')

            return total_score

        optimizer.step(closure)

    # a last correction...
    input_param.data.clamp_(0, 1)

    # print(cur_nEpochs == min_nEpochs)

    # make nEpochs divisible by 20 to avoid overtraining by closure
    min_nEpochs[0] -= min_nEpochs[0] % 20

    if findMin:
        out.write('Optimum epochs: %d\n\n' % min_nEpochs[0])
        out.close()
        # overtrained....need to redo learning up until optimal number of epochs
        if not (num_steps == min_nEpochs[0]):
            return run_style_transfer(copies['cnn'],
                                      copies['content_img'],
                                      copies['style_img'],
                                      copies['input_img'],
                                      outfile,
                                      num_steps=min_nEpochs[0], findMin=False)

    return input_param.data

######################################################################
# Finally, run the algorithm

def main():
    cnn = models.vgg19(pretrained=True).features
    # move it to the GPU if possible:
    if use_cuda:
        cnn = cnn.cuda()

    for i, content in enumerate(content_style_dict.keys()):
        c_folder = os.path.join(results_dir, content)
        if not os.path.exists(c_folder):
            os.makedirs(c_folder)

        print('\n\nTransforming %s.....(%d / %d)' % (content, i+1, len(content_style_dict)))

        content_img = image_loader(os.path.join(content_dir,'%s.jpg' % content)).type(dtype)
        imsave(content_img.data, os.path.join(c_folder, 'content.jpg'))

        for j, style in enumerate(content_style_dict[content]):
            s_folder = os.path.join(c_folder, style)
            if not os.path.exists(s_folder):
                os.mkdir(s_folder)

            style_img = image_loader(os.path.join(styles_dir,'%s.jpg' % style)).type(dtype)
            assert style_img.size() == content_img.size(), \
                "we need to import style and content images of the same size"

            print('\n.....Using %s style......(%d / %d)\n' % (style, j+1, len(content_style_dict[content])))
            # save content again at this level
            imsave(content_img.data, os.path.join(s_folder, 'content.jpg'))
            imsave(style_img.data, os.path.join(s_folder,'style.jpg'))


            input_img = content_img.clone()
            # if you want to use a white noise instead uncomment the below line:
            # input_img = Variable(torch.randn(content_img.data.size())).type(dtype)

            # add the original input image to the figure:
            # plt.figure()
            # imshow(input_img.data, title='Input Image')
            # imsave(input_img.data, 'input.jpg')

            outfile = os.path.join(s_folder,'records.txt')

            output = run_style_transfer(cnn, content_img, style_img, input_img, outfile)

            # plt.figure()
            # imshow(output, title='Output Image')
            imsave(output, os.path.join(s_folder,'output.jpg'))

            # sphinx_gallery_thumbnail_number = 4
            # plt.ioff()
            # plt.show()

if __name__ == '__main__':
    main()
