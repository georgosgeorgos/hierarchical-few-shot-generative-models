from model.vae import VAE

from model.ns import NS
from model.tns import TNS

from model.hfsgm import HFSGM
from model.thfsgm import THFSGM

from model.cns import CNS
from model.ctns import CTNS

from model.chfsgm import CHFSGM
from model.cthfsgm import CTHFSGM
from model.chfsgm_multiscale import CHFSGM_MULTISCALE

#from model.cns_res import CNS_res

def select_model(args):
    if args.model == "vae":
        return VAE    
    elif args.model == "ns":
        return NS
    elif args.model == "tns":
        return TNS
    
    elif args.model == "hfsgm":
        return HFSGM
    elif args.model == "thfsgm":
        return THFSGM
    
    elif args.model == "cns":
        return CNS
    elif args.model == "ctns":
        return CTNS
    
    elif args.model == "chfsgm":
        return CHFSGM
    elif args.model == "cthfsgm":
        return CTHFSGM
    elif args.model == "chfsgm_multiscale":
        return CHFSGM_MULTISCALE
    else:
        print("No valid model selected. Please choose {vae, ns, tns, hfsgm, thfsgm, hfsg_ar}")
