import torch


def remove_outliers_by_sensitivity(
    model,
    gradients,
    sensitivity,
):
    module_names = list(model.keys())
    outlier_weights = []
    total_outliers = 0
    total_weights = 0

    def _body(gweight, weight):
        num_outliers = int(gweight.numel() * sensitivity / 100)
        thres = gweight.reshape(-1).topk(k=num_outliers).values[-1]

        t = gweight > thres

        outlier_weight = weight * t
        weight = weight * ~t
        # print((weight == 0).sum().item() / weight.numel())
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for _name in module_names:
        weight = model[_name].to(torch.float)
        gweight = gradients[_name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            gweight, weight
        )
        model[_name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        outlier_weights.append(outlier_weight)

    print("p outlier:", total_outliers / total_weights * 100)
    return [outlier_weights]


def remove_outliers_by_threshold(
    model,
    outlier_config,
    outlier_weights=None,  # to avoid memory leak
):
    module_names = list(model.keys())
    if outlier_weights is None:
        outlier_weights = [[0 for _ in range(len(module_names))]]

    total_outliers = 0
    total_weights = 0

    def _body(weight, thres):
        t = torch.logical_or(
            weight >= thres,
            weight <= -thres,
        )

        outlier_weight = weight * t
        weight = weight * ~t
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for i, name in enumerate(module_names):
        thres = outlier_config[name]
        weight = model[name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            weight, thres
        )
        model[name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        # import pdb; pdb.set_trace()
        outlier_weights[0][i] += outlier_weight

    print("p outlier:", total_outliers / total_weights * 100)
    return outlier_weights


def remove_outliers(
    model,
    sensitivity,
    outlier_config,
    gradients=None,
):
    # model and gradients are dictionary of a layer component
    # where the key is the layer name (e.g. q, k, v) and the value is the weight
    assert isinstance(model, dict)
    assert isinstance(gradients, dict) or gradients is None

    assert outlier_config is not None or sensitivity != 0
    if sensitivity != 0:
        assert gradients is not None

    if sensitivity != 0:
        print("removing outliers by sensitivity")
        outlier_weights = remove_outliers_by_sensitivity(
            model=model,
            gradients=gradients,
            sensitivity=sensitivity,
        )
    else:
        outlier_weights = None

    if outlier_config is not None:
        print("removing outliers by threshold")
        outlier_weights = remove_outliers_by_threshold(
            model=model,
            outlier_config=outlier_config,
            outlier_weights=outlier_weights,
        )

    return outlier_weights
