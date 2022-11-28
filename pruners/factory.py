from pruners.HSIClasso import HSIC_lasso_pruning

pruner_methods = \
{
    "HSIC_lasso": HSIC_lasso_pruning
}


def get_pruner(name_str):
    if name_str in pruner_methods.keys():
        return pruner_methods[name_str]
    else:
        print("pruner {} is not supported".format(name_str))
        return None