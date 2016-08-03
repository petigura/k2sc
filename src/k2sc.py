#!/usr/bin/env python
from __future__ import print_function, division
print("hello")
import os
import sys
import errno
import warnings
import logging

import math as mt
import numpy as np
import scipy as sp
import pyfits as pf
import matplotlib.pyplot as pl

from copy import copy   
from collections import namedtuple
from matplotlib.backends.backend_pdf import PdfPages
from numpy import (any, array, ones_like, fromstring, tile, median,
                   zeros_like, exp, isfinite, nanmean, argmax, argmin)
from numpy.random import normal

from time import time, sleep
from datetime import datetime
from os.path import join, exists, abspath, basename
from argparse import ArgumentParser

from core import *
from detrender import Detrender
from kernels import kernels, BasicKernel, BasicKernelEP, QuasiPeriodicKernel, QuasiPeriodicKernelEP
from k2io import select_reader, FITSWriter, SPLOXReader, MASTPixelReader
from cdpp import cdpp
from de import DiffEvol
from ls import fasper
from utils import medsig

import k2phot
import k2phot.pipeline_k2sc

warnings.resetwarnings()
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=DeprecationWarning, append=True)

##TODO: Copy the FITS E0 header if a fits file is available
##TODO: Define the covariance matrix splits for every campaign

mpi_root = 0
mpi_rank = 1

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(name)s: %(message)s')

def nanmedian(a):
    return np.median(a[isfinite(a)])


def psearch(time, flux, min_p, max_p):
    freq,power,nout,jmax,prob = fasper(time, flux, 6, 0.5)
    period = 1/freq
    m = (period > min_p) & (period < max_p) 
    period, power = period[m], power[m]
    j = argmax(power)

    expy = mt.exp(-power[j])
    effm = 2*nout/6
    fap  = expy*effm

    if fap > 0.01:
        fap = 1.0-(1.0-expy)**effm
    
    return period[j], fap

def detrend(dataset, args):
    """
    Needs to have args defined
    """

    ## Setup the logger
    ## ----------------
    logger  = logging.getLogger('Worker %i' % mpi_rank)
    logger.name = '<{:d}>'.format(dataset.epic)

    np.seterrcall(lambda e,f: logger.info(e))
    np.seterr(invalid='ignore')

    ## Main variables
    ## --------------
    Result  = namedtuple('SCResult', 'detrender pv tr_time tr_position cdpp_r cdpp_t cdpp_c warn')
    results = []  # a list of Result tuples, one per aperture
    masks   = []  # a list of light curve masks, one per aperture 

    ## Initialise utility variables
    ## ----------------------------
    ds   = dataset
    info = logger.info

    ## Periodic signal masking
    ## -----------------------
    if args.p_mask_center and args.p_mask_period and args.p_mask_duration:
        ds.mask_periodic_signal(
            args.p_mask_center, args.p_mask_period, args.p_mask_duration
            )

    ## Initial outlier and period detection
    ## ------------------------------------
    ## We carry out an initial outlier and period detection using
    ## a default GP hyperparameter vector based on campaign 4 fits
    ## done using (almost) noninformative priors.

    for iset in range(ds.nsets):
        flux = ds.fluxes[iset]
        inputs = np.transpose([ds.time,ds.x,ds.y])
        detrender = Detrender(
            flux, inputs, mask=isfinite(flux), splits=args.splits, 
            kernel=BasicKernelEP(), tr_nrandom=args.tr_nrandom,
            tr_nblocks=args.tr_nblocks, tr_bspan=args.tr_bspan
            )
    
        ttrend,ptrend = detrender.predict(
            detrender.kernel.pv0+1e-5, components=True
            )

        cflux = flux - ptrend + median(ptrend) - ttrend + median(ttrend)
        cflux /= nanmedian(cflux)

        ## Iterative sigma-clipping
        ## ------------------------
        info('Starting initial outlier detection')
        fmask  = isfinite(cflux)
        omask  = fmask.copy()
        i, nm  = 0, None
        while nm != omask.sum() and i<10:
            nm = omask.sum()
            _, sigma = medsig(cflux[omask])
            omask[fmask] &= (cflux[fmask] < 1+5*sigma) & (cflux[fmask] > 1-5*sigma)
            i += 1
        masks.append(fmask)
        ofrac = (~omask).sum() / omask.size
        if ofrac < 0.25:
            masks[-1] &= omask
            info('  Flagged %i (%4.1f%%) outliers.', (~omask).sum(), ofrac)
        else:
            info('  Found %i (%4.1f%%) outliers. Not flagging..', (~omask).sum(), ofrac)

        ## Lomb-Scargle period search
        ## --------------------------
        info('Starting Lomb-Scargle period search')
        mask  = masks[-1]
        nflux = flux - ptrend + nanmedian(ptrend)
        ntime = ds.time - ds.time.mean()
        pflux = np.poly1d(np.polyfit(ntime[mask], nflux[mask], 9))(ntime)
        period, fap = psearch(ds.time[mask], (nflux-pflux)[mask], args.ls_min_period, args.ls_max_period)
        
        if fap < 1e-50:
            ds.is_periodic = True
            ds.ls_fap    = fap
            ds.ls_period = period
        
    ## Kernel selection
    ## ----------------
    args.kernel='basic'
    if args.kernel:
        info('Overriding automatic kernel selection, using %s kernel as given in the command line', args.kernel)
        if 'periodic' in args.kernel and not args.kernel_period:
            logger.critical('Need to give period (--kernel-period) if overriding automatic kernel detection with a periodic kernel. Quitting.')
            exit(1)
        kernel = kernels[args.kernel](period=args.kernel_period)
    else:
        info('  Using %s position kernel', args.default_position_kernel)
        if ds.is_periodic:
            info('  Found periodicity p = {:7.2f} (fap {:7.4e} < 1e-50), will use a quasiperiodic kernel'.format(ds.ls_period, ds.ls_fap))
        else:
            info('  No strong periodicity found, using a basic kernel')

        if args.default_position_kernel.lower() == 'sqrexp':
            kernel = QuasiPeriodicKernel(period=ds.ls_period)   if ds.is_periodic else BasicKernel() 
        else:
            kernel = QuasiPeriodicKernelEP(period=ds.ls_period) if ds.is_periodic else BasicKernelEP()


    ## Detrending
    ## ----------
    for iset in range(ds.nsets):
        if ds.nsets > 1:
            logger.name = 'Worker {:d} <{:d}-{:d}>'.format(mpi_rank, dataset.epic, iset+1)
        np.random.seed(args.seed)
        tstart = time()
        inputs = np.transpose([ds.time,ds.x,ds.y])
        detrender = Detrender(ds.fluxes[iset], inputs, mask=masks[iset],
                              splits=args.splits, kernel=kernel, tr_nrandom=args.tr_nrandom,
                              tr_nblocks=args.tr_nblocks, tr_bspan=args.tr_bspan)

        de = DiffEvol(detrender.neglnposterior, kernel.bounds, args.de_npop)

        ## Period population generation
        ## ----------------------------
        if isinstance(kernel, QuasiPeriodicKernel):
            de._population[:,2] = np.clip(normal(kernel.period, 0.1*kernel.period, size=de.n_pop),
                                          args.ls_min_period, args.ls_max_period)

        ## Global hyperparameter optimisation
        ## ----------------------------------
        info('Starting global hyperparameter optimisation using DE')
        tstart_de = time()
        for i,r in enumerate(de(args.de_niter)):
            info('  DE iteration %3i -ln(L) %4.1f', i, de.minimum_value)
            tcur_de = time()
            if ((de._fitness.ptp() < 3) or (tcur_de - tstart_de > args.de_max_time)) and (i>2):
                break
        info('  DE finished in %i seconds', tcur_de-tstart_de)
        info('  DE minimum found at: %s', np.array_str(de.minimum_location, precision=3, max_line_width=250))
        info('  DE -ln(L) %4.1f', de.minimum_value)

        ## Local hyperparameter optimisation
        ## ---------------------------------
        info('Starting local hyperparameter optimisation')
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
                pv, warn = detrender.train(de.minimum_location)
        except ValueError as e:
            logger.error('Local optimiser failed, %s', e)
            logger.error('Skipping the file')
            return
        info('  Local minimum found at: %s', np.array_str(pv, precision=3))

        ## Trend computation
        ## -----------------
        (mt,tt),(mp,tp) = map(lambda a: (nanmedian(a), a-nanmedian(a)), detrender.predict(pv, components=True))

        ## Iterative sigma-clipping
        ## ------------------------
        info('Starting final outlier detection')
        flux = detrender.data.unmasked_flux
        cflux = flux-tp-tt
        cflux /= nanmedian(cflux)

        fmask = isfinite(cflux)
        mhigh = zeros_like(fmask)
        mlow  = zeros_like(fmask)
        mask  = fmask.copy()
        i, nm = 0, None
        while nm != mask.sum() and i<10:
            nm = mask.sum()
            _, sigma = medsig(cflux[mask])
            mhigh[fmask] = cflux[fmask] > 1+5*sigma
            mlow[fmask]  = cflux[fmask] < 1-5*sigma
            mask &= fmask & (~mlow) & (~mhigh)
            i += 1
        ds.mflags[iset][~fmask] |= M_NOTFINITE
        ds.mflags[iset][mhigh]  |= M_OUTLIER_U
        ds.mflags[iset][mlow]   |= M_OUTLIER_D
        
        info('  %5i too high', mhigh.sum())
        info('  %5i too low',  mlow.sum())
        info('  %5i not finite', (~fmask).sum())

        ## Detrending and CDPP computation
        ## -------------------------------
        info('Computing time and position trends')
        dd = detrender.data
        cdpp_r = cdpp(dd.masked_time,   dd.masked_flux)
        cdpp_t = cdpp(dd.unmasked_time, dd.unmasked_flux-tp,    exclude=~dd.mask)
        cdpp_c = cdpp(dd.unmasked_time, dd.unmasked_flux-tp-tt, exclude=~dd.mask)
        results.append(Result(detrender, pv, tt+mt, tp+mp, cdpp_r, cdpp_t, cdpp_c, warn))
        info('  CDPP - raw - %6.3f', cdpp_r)
        info('  CDPP - position component removed - %6.3f', cdpp_t)
        info('  CDPP - full reduction - %6.3f', cdpp_c)
        info('Detrending time %6.3f', time()-tstart)

    info('Finished')
    return dataset, results

class Object(object):
    pass

def main(dataset, splits, queit=True, seed=0,
         default_position_kernel='SqrExp', kernel=None,
         kernel_period=None, p_mask_center=None, p_mask_duration=None,
         tr_nrandom=400, tr_nblocks=6, tr_bspan=50, de_npop=100,
         de_niter=150, de_max_time=300.0, ls_max_fap=-50.0,
         ls_min_period=0.05, ls_max_period=25.0, flare_sigma=5.0,
         flare_erosion=5, outlier_sigma=5.0, outlier_mwidth=25):
    """Emulation of __main__ from bin/k2sc, generates a stand in for the args
    object which is passed around.

    :param splits:
    :type splits: list of ints
    
    :param quiet:
    :type quiet: suppress messages

    :param seed: initialized DE (default 0)
    :type seed: int 

    :param default_position_kernel: choices=['SqrExp','Exp'], default=SqrExp
    :type default_position_kernel: str 

    :param kernel:
    :type kernel: str 

    :param kernel_period: (default None)
    :type kernel_period: float 

    :param p_mask_center: (default None)
    :type p_mask_center: float

    :param p_mask_duration: default None
    :type p_mask_duration: float

    :param tr_nrandom: Number of random samples (default 400)
    :type tr_nrandom: int

    :param tr_nblocks: Number of sample blocks (default 6) 
    :type param: int

    :param tr_bspan: Span of a single block (default 50)
    :type tr_bspan: int

    :param de_npop: 
        Size of the differential evolution parameter
        vector population (default 100)
    :type de_npop: int

    :param de_niter: Number of differential evolution iterations (default 150)
    :type de_niter: int

    :param de_max_time:
        Maximum time used for differential evolution (default 300)
    :type de_max_time: float 
    
    :param ls_max_fap: 
        Maximum Lonb-Scargle log10(false alarm) threshold to use
        periodic kernel (default -50.0) 
    :type ls_max_fap: float

    :param ls_min_period: Minimum period to search for (default 0.05)
    :type ls_min_period: float

    :param ls_max_period: Maximum period to search for (default 25)
    :type ls_max_period: float

    :param flare_sigma: (default 5.0)
    :type flare_sigma: float
    
    :param flare_erosion: (default 5)
    :type flare_erosion: int

    :param outlier_sigma: (default 5.0)
    :type outlier_sigma: float
    
    :param outlier_mwidth: (default 25)
    :type outlier_mwidth: int

    Examples

    bin/k2sc -c 3 /Users/petigura//Research/K2//k2_archive/pixel/C3/ktwo205904628-c03_lpd-targ.fits

    """
    args = Object()
    args.dataset = dataset
    args.splits = splits
    args.queit = queit
    args.seed = seed
    args.kernel = kernel
    args.default_position_kernel = default_position_kernel
    args.kernel_period = kernel_period
    args.p_mask_center = p_mask_center
    args.p_mask_duration = p_mask_duration
    args.tr_nrandom = tr_nrandom
    args.tr_nblocks = tr_nblocks
    args.tr_bspan = tr_bspan
    args.de_npop = de_npop
    args.de_niter  = de_niter
    args.de_max_time = de_max_time
    args.ls_max_fap = ls_max_fap
    args.ls_min_period = ls_min_period
    args.ls_max_period = ls_max_period
    args.flare_sigma = flare_sigma
    args.flare_erosion = flare_erosion
    args.outlier_sigma = outlier_sigma
    args.outlier_mwidth = outlier_mwidth

#    csplits = {4: [2240,2273], 3: [2180], 1:[2017,2022] }
#    transfn = {1:'pixeltrans_C1_ch04.h5' , 3: 'pixeltrans_C3_ch04.h5',}
#    transfn = transfn[args.campaign]
#    transfn = join('/Users/petigura//Research/K2//k2photfiles/',transfn)
#    print(args)

    ## Logging
    logger = logging.getLogger('Master')

    if args.splits is None:
        splits = csplits[args.campaign]
    else:
        splits = args.splits


    ## Select data reader
    ## NOTE: We don't allow mixed input types per run

    logger.info('  Differential evolution parameters')
    logger.info('  Population size: {:3d}'.format(args.de_npop))
    logger.info('  Number of iterations: {:3d}'.format(args.de_niter))
    logger.info('  Maximum DE time: {:6.2f} seconds'.format(args.de_max_time))
    logger.info('')

    results = detrend(dataset, args)
    return results

