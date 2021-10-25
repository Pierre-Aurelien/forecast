"""Module to perform statistical inference."""
import numdifftools as nd
import numpy as np
import pandas as pd
import scipy.stats as stats
from joblib import Parallel, delayed
from scipy.optimize import minimize


def starting_point(i, experiment):
    """Computes the starting point for ML minimisation.

    Args:
        i: construct number
        experiment: instance of experiment class

    Returns:
        Empirical mean and variance
    """
    t = np.ceil(experiment.nijhat[i, :]).astype(int)
    t = np.repeat(experiment.mean_assigned, t)
    if np.max(t) == np.min(t):  # What if all the cells fall into one unique bin?
        j = np.where(experiment.mean_assigned == np.max(t))[0][0]
        mu = np.max(t)
        std = (experiment.partitioning[j + 1] - experiment.partitioning[j]) / 4
    elif not np.any(t):
        return np.array([0, 0])
    else:
        mu = np.mean(t)
        std = np.std(t, ddof=1)
    return np.array([mu, std ** 2])


def neg_ll_rep(theta, i, experiment):  # noqa: CCR001
    """Computes the likelihood for a set of parameters.

    Args:
        theta: logparameter theta=(alpha,beta)
        i: construct number
        experiment: instance of experiment class

    Returns:
        negative log likelihood
    """
    alpha = theta[0]
    beta = theta[1]
    nl = 0
    for j in range(experiment.bins):
        if experiment.nj[j] == 0:
            pass
        else:
            # Compute intensity parameter
            if experiment.distribution == "lognormal":
                probability_bin = stats.norm.cdf(
                    experiment.partitioning[j + 1], loc=np.exp(alpha), scale=np.exp(beta)
                ) - stats.norm.cdf(
                    experiment.partitioning[j], loc=np.exp(alpha), scale=np.exp(beta)
                )
            else:
                probability_bin = stats.gamma.cdf(
                    experiment.partitioning[j + 1], a=np.exp(alpha), scale=np.exp(beta)
                ) - stats.gamma.cdf(experiment.partitioning[j], a=np.exp(alpha), scale=np.exp(beta))
            intensity = (
                experiment.nihat[i] * probability_bin * experiment.reads[j] / experiment.nj[j]
            )
            # Compute Likelihood
            if experiment.sequencing[i, j] != 0:
                if intensity > 0:  # Avoid float error with np.log
                    nl += intensity - experiment.sequencing[i, j] * np.log(intensity)
            else:
                nl += intensity
    return nl


def neg_ll(theta, i, experiment):  # noqa: CCR001
    """Computes the likelihood for a set of parameters.

    Args:
        theta: parameter theta=(alpha,beta)
        i: construct number
        experiment: instance of experiment class

    Returns:
        negative log likelihood
    """
    alpha = theta[0]
    beta = theta[1]
    nl = 0
    for j in range(experiment.bins):
        if experiment.nj[j] == 0:
            pass
        else:
            # Compute intensity parameter
            if experiment.distribution == "lognormal":
                probability_bin = stats.norm.cdf(
                    experiment.partitioning[j + 1], loc=alpha, scale=beta
                ) - stats.norm.cdf(experiment.partitioning[j], loc=alpha, scale=beta)
            else:
                probability_bin = stats.gamma.cdf(
                    experiment.partitioning[j + 1], a=alpha, scale=beta
                ) - stats.gamma.cdf(experiment.partitioning[j], a=alpha, scale=beta)
            intensity = (
                experiment.nihat[i] * probability_bin * experiment.reads[j] / experiment.nj[j]
            )
            # Compute Likelihood
            if experiment.sequencing[i, j] != 0:
                if intensity > 0:  # Avoid float error with np.log
                    nl += intensity - experiment.sequencing[i, j] * np.log(intensity)
            else:
                nl += intensity
    return nl


def reparameterised_ml_inference_(i, experiment) -> np.ndarray:  # noqa: CCR001
    """Conduct inference for construct i based on experimental data.

    Args:
        i: construct number
        experiment: instance of experiment class

    Returns:
        Returns ML inference,confidence intervals, MOM inference,
        scoring and validity of ML inference
    """
    data_results = np.zeros(8)
    t = experiment.nijhat[i, :]
    if np.sum(t) != 0:  # Can we do inference? has the genetic construct been sequenced?
        data_results[7] = (t[0] + t[-1]) / np.sum(
            t
        )  # Scoring of the data- How lopsided is the read count? all on the left-right border?
        sp = starting_point(i, experiment)
        # the four next lines provide the MOM estimates on a,b, mu and sigma
        if experiment.distribution == "lognormal":
            data_results[4] = sp[0]  # value of mu MOM
            data_results[5] = np.sqrt(sp[1])  # Value of sigma MOM
            if (
                np.count_nonzero(t) == 1
            ):  # is there only one bin to be considered? then naive inference
                data_results[6] = 3  # Inference grade 3 : Naive inference

            else:  # in the remaining case, we can deploy the mle framework to improve the mom estimation
                rep_sp = np.log(
                    np.array([sp[0], np.sqrt(sp[1])])
                )  # initial value for log(mu,sigma)
                res = minimize(neg_ll_rep, rep_sp, args=(i, experiment), method="Nelder-Mead")
                c, d = res.x
                data_results[0], data_results[1] = np.exp(c), np.exp(d)
                # Compute confidence intervals with hessian

                def fi(x):
                    neg_ll(x, i, experiment)

                fdd = nd.Hessian(fi)
                hessian_ndt = fdd([np.exp(c), np.exp(d)])
                if np.isfinite(hessian_ndt).all():
                    with np.errstate(invalid="ignore"):
                        if np.all(np.linalg.eigvals(hessian_ndt) > 0):
                            inv_jacobian = np.linalg.inv(hessian_ndt)
                            if experiment.distribution == "lognormal":
                                jacobian = np.diag((1, 1))
                            e, f = np.sqrt(
                                np.diag(np.matmul(np.matmul(jacobian, inv_jacobian), jacobian.t))
                            )
                            data_results[2] = e
                            data_results[3] = f
                            data_results[6] = 1  # Inference grade 1 : ML inference  successful

                        else:
                            data_results[
                                6
                            ] = 2  # Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
                else:
                    data_results[6] = 2
        elif experiment.distribution == "gamma":
            data_results[4] = sp[0]  # value of mu MOM
            data_results[5] = np.sqrt(sp[1])  # Value of sigma MOM
            if (
                np.count_nonzero(t) == 1
            ):  # is there only one bin to be considered? then naive inference
                data_results[6] = 3  # Inference grade 3 : Naive inference

            else:  # in the remaining case, we can deploy the mle framework to improve the mom estimation
                rep_sp = np.log(
                    np.array([(sp[0] ** 2) / sp[1], (sp[1]) / sp[0]])
                )  # initial value for log(a,b)
                res = minimize(neg_ll_rep, rep_sp, args=(i, experiment), method="Nelder-Mead")
                c, d = res.x
                data_results[0], data_results[1] = np.exp(c), np.exp(d)
                # Compute confidence intervals with hessian

                def fi(x):
                    neg_ll(x, i, experiment)

                fdd = nd.Hessian(fi)
                hessian_ndt = fdd([np.exp(c), np.exp(d)])
                if np.isfinite(hessian_ndt).all():
                    with np.errstate(invalid="ignore"):
                        if np.all(np.linalg.eigvals(hessian_ndt) > 0):
                            inv_jacobian = np.linalg.inv(hessian_ndt)
                            jacobian = np.diag(
                                (1, 1)
                            )  # np.array([[b,a],[b/(2*np.sqrt(a)),np.sqrt(a)]])
                            e, f = np.sqrt(
                                np.diag(np.matmul(np.matmul(jacobian, inv_jacobian), jacobian.T))
                            )
                            data_results[2] = e
                            data_results[3] = f
                            data_results[6] = 1  # Inference grade 1 : ML inference  successful

                        else:
                            data_results[
                                6
                            ] = 2  # Inference grade 2 : ML inference, although the hessian is not inverstible at the minimum... Probably an issue with the data and model mispecification
                else:
                    data_results[6] = 2
    else:
        data_results[6] = 4  # Inference grade 4: No inference is possible
    return data_results


def inference(p, q, experiment):
    """Wrapper for parallel inference.

    Args:
        p: first construct
        q: last construct
        experiment: instance of the experiment class

    Returns:
        inference results for p-q+1 constructs
    """
    data_results = Parallel(n_jobs=-1, max_nbytes=None)(
        delayed(reparameterised_ml_inference_)(i, experiment) for i in range(p, q)
    )
    data_results = np.array(data_results)
    df = pd.DataFrame(data_results)
    if experiment.distribution == "lognormal":
        df.rename(
            columns={
                0: "mu_MLE",
                1: "sigma_MLE",
                2: "mu_std",
                3: "sigma_std",
                4: "mu_MOM",
                5: "sigma_MOM",
                6: "Inference_grade",
                7: "Score",
            },
            errors="raise",
            inplace=True,
        )
    elif experiment.distribution == "gamma":
        df.rename(
            columns={
                0: "a_MLE",
                1: "b_MLE",
                2: "a_std",
                3: "b_std",
                4: "mu_MOM",
                5: "sigma_MOM",
                6: "Inference_grade",
                7: "Score",
            },
            errors="raise",
            inplace=True,
        )
    return df
