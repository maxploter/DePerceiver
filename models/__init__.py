# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build
from .perciever import build as build_perceiver


def build_model(args):
    if args.model == 'perceiver':
        return build_perceiver(args)
    else:
        return build(args)
