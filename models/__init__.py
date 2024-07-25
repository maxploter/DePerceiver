# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .perceiver import build as build_perceiver
from .perceiver_io import build as build_perceiver_io


def build_model(args):
    if args.model == 'perceiver':
        return build_perceiver(args)
    elif args.model == 'perceiver_io':
        return build_perceiver_io(args)
    else:
        return build(args)
