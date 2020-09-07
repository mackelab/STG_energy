import numpy as np
from copy import deepcopy
import torch

def buildCondCovMatrix(posterior, lims, samples=None, num_samples=10, max_dim=None, resolution=20):
    """Build the full conditional correlation matrix."""

    if max_dim is None:
        max_dim = posterior.ndim
    if samples is None:
        samples = posterior.sample(num_samples).detach().numpy()

    mdnewcounter = 0
    all_rho_images = []
    for theta in samples:
        rho_image_11 = np.zeros((max_dim, max_dim))
        for d1 in range(max_dim):
            for d2 in range(max_dim):
                if d1 < d2:
                    p_image = eval_conditional_density(posterior, [theta], lims, dim1=d1, dim2=d2,
                                                       resolution=resolution, log=False)
                    cc = conditional_correlation(p_image, lims[d1,0], lims[d1,1], lims[d2,0], lims[d2,1])
                    cc.calc_rhoXY()
                    rho_image_11[d1, d2] = cc.rho
        all_rho_images.append(rho_image_11)
        mdnewcounter += 1

    mean_conditional_correlation = np.nanmean(all_rho_images, axis=0)

    rho_symm = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        for d2 in range(max_dim):
            if d1 < d2:
                rho_symm[d1, d2] = mean_conditional_correlation[d1, d2]
            elif d1 > d2:
                rho_symm[d1, d2] = mean_conditional_correlation[d2, d1]
            else:
                rho_symm[d1, d2] = 1.0

    return rho_symm


def extractSpecificCondCorr(posterior, corrs, lims, samples=None, num_samples=10, max_dim=None, resolution=20):
    """Find the conditional correlation between any two parameters."""

    if max_dim is None:
        max_dim = posterior.ndim
    if samples is None:
        samples = posterior.gen(num_samples) # TODO: rejection sampling!

    all_lists_of_corrs = []
    for theta in samples:
        list_of_corrs = []

        for pair in corrs:
            d1 = pair[0]
            d2 = pair[1]
            if d1 < d2:
                p_image = eval_conditional_density(posterior, [theta], lims, dim1=d1, dim2=d2,
                                                   resolution=resolution, log=False)
                cc = conditional_correlation(p_image, lims[d1,0], lims[d1,1], lims[d2,0], lims[d2,1])
                cc.calc_rhoXY()
                list_of_corrs.append(cc.rho)
        all_lists_of_corrs.append(list_of_corrs)

    all_lists_of_corrs = np.asarray(all_lists_of_corrs).T

    return all_lists_of_corrs


def eval_conditional_density(pdf, theta, lims, dim1, dim2, resolution=20, log=True):
    """Evaluate the conditional plane."""

    print("yesa")
    print("lims", lims)
    print("theta", theta)

    if dim1 == dim2:
        gbar_dim1 = np.linspace(lims[dim1,0], lims[dim1,1], resolution)

        p_vector = np.zeros(resolution)

        list_of_current_point_eval = []

        for index_gbar1 in range(resolution):
            current_point_eval = deepcopy(theta)[0]
            current_point_eval[dim1] = gbar_dim1[index_gbar1]
            list_of_current_point_eval.append(current_point_eval)

        var =  torch.cat(list_of_current_point_eval).reshape((resolution, 31))
        p = pdf.log_prob(var)

        for index_gbar1 in range(resolution):
            p_vector[index_gbar1] = p[index_gbar1]

        if log:
            return p_vector
        else:
            return np.exp(p_vector)
    else:
        gbar_dim1 = np.linspace(lims[dim1,0]+1e-5, lims[dim1,1]-1e-5, resolution) # todo get rid of 1e-5 (which is there to avoid being outside of the prior bounds)
        gbar_dim2 = np.linspace(lims[dim2,0]+1e-5, lims[dim2,1]-1e-5, resolution)

        p_image = np.zeros((resolution, resolution))
        list_of_current_point_eval = []

        for index_gbar1 in range(resolution):
            for index_gbar2 in range(resolution):
                current_point_eval = deepcopy(theta)[0]
                current_point_eval[dim1] = gbar_dim1[index_gbar1]
                current_point_eval[dim2] = gbar_dim2[index_gbar2]
                list_of_current_point_eval.append(torch.as_tensor(current_point_eval))

        var =  torch.cat(list_of_current_point_eval).reshape((resolution**2, 31))

        p = pdf.log_prob(var)
        i = 0
        for index_gbar1 in range(resolution):
            for index_gbar2 in range(resolution):
                p_image[index_gbar1, index_gbar2] = p[i]
                i += 1
        if log:
            return p_image
        else:
            return np.exp(p_image)


class conditional_correlation:
    def __init__(self, cPDF, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(cPDF)
        self.pdfXY = self.normalize_pdf_2D(cPDF)
        self.pdfX  = None
        self.pdfY  = None
        self.EX    = None
        self.EY    = None
        self.EXY   = None
        self.VarX  = None
        self.VarY  = None
        self.CovXY = None
        self.rho   = None

    @staticmethod
    def normalize_pdf_1D(pdf, lower, upper, resolution):
        return pdf * resolution / (upper - lower) / np.sum(pdf)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_rhoXY(self):
        self.calc_marginals()
        self.calc_EXY()
        self.EX  = conditional_correlation.calc_E_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.EY = conditional_correlation.calc_E_1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.CovXY = self.EXY - self.EX * self.EY
        self.VarX = conditional_correlation.calc_var1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.VarY = conditional_correlation.calc_var1D(self.pdfY, self.ly, self.uy, self.resolution_y)
        self.rho = self.CovXY / np.sqrt(self.VarX * self.VarY)

    def calc_EXY(self):
        x_matrix = np.tile(np.linspace(self.lx, self.ux, self.resolution_x), (self.resolution_y, 1))
        y_matrix = np.tile(np.linspace(self.ly, self.uy, self.resolution_y), (self.resolution_x, 1)).T
        self.EXY = np.sum(  np.sum(  x_matrix * y_matrix * self.pdfXY  )  )
        self.EXY /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)

    @staticmethod
    def calc_E_1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E = np.sum(x_vector * pdf)
        E /= resolution / (upper - lower)
        return E

    @staticmethod
    def calc_var1D(pdf, lower, upper, resolution):
        x_vector = np.linspace(lower, upper, resolution)
        E2 = np.sum(x_vector**2 * pdf)
        E2 /= resolution / (upper - lower)
        var = E2 - conditional_correlation.calc_E_1D(pdf, lower, upper, resolution)**2
        return var

    def calc_marginals(self):
        self.pdfX = np.sum(self.pdfXY, axis=0)
        self.pdfY = np.sum(self.pdfXY, axis=1)

        self.pdfX = conditional_correlation.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.pdfY = conditional_correlation.normalize_pdf_1D(self.pdfY, self.ly, self.uy, self.resolution_y)




def partialCorrelation(posterior_samples, max_dim):
    import sklearn.linear_model as lm
    import scipy.stats

    rho_image_11 = np.zeros((max_dim, max_dim))
    significance_matrix = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        print('Starting new dimension:  ', d1)
        for d2 in range(max_dim):
            if d1 < d2:
                vec = np.ones(max_dim, dtype=int)
                vec[d1] = 0
                vec[d2] = 0
                condition_samples = posterior_samples[:,vec]
                samplesX = posterior_samples[:,d1]
                samplesY = posterior_samples[:,d2]
                regX = lm.LinearRegression().fit(condition_samples, samplesX)
                regY = lm.LinearRegression().fit(condition_samples, samplesY)

                predictedX = regX.predict(condition_samples)
                predictedY = regY.predict(condition_samples)

                residualsX = samplesX - predictedX
                residualsY = samplesY - predictedY

                partial_corr, p_val = scipy.stats.pearsonr(residualsX, residualsY)
                rho_image_11[d1, d2]        = partial_corr
                significance_matrix[d1, d2] = p_val

    rho_symm = np.zeros((max_dim, max_dim))
    for d1 in range(max_dim):
        for d2 in range(max_dim):
            if d1 < d2:
                rho_symm[d1, d2] = rho_image_11[d1, d2]
            elif d1 > d2:
                rho_symm[d1, d2] = rho_image_11[d2, d1]
            else:
                rho_symm[d1, d2] = 1.0

    return rho_symm, significance_matrix



class conditional_mutual_information:
    def __init__(self, cPDF, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(cPDF)
        self.pdfXY = self.normalize_pdf_2D(cPDF)
        self.pdfX  = None
        self.pdfY  = None
        self.HX    = None
        self.IXY   = None

    @staticmethod
    def normalize_pdf_1D(pdf, lower, upper, resolution):
        return pdf * resolution / (upper - lower) / np.sum(pdf)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_HX(self):
        self.pdfX = self.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.HX = -np.sum(self.pdfX * np.log(self.pdfX))
        self.HX /= self.resolution_x / (self.ux - self.lx)

    def calc_IXY(self):
        self.calc_marginals()
        pdfXmatrix = np.tile(self.pdfX, (self.resolution_y,1))
        pdfYmatrix = np.tile(self.pdfY, (self.resolution_x,1)).T
        self.IXY   = np.sum(self.pdfXY * np.log(self.pdfXY / pdfXmatrix / pdfYmatrix))
        self.IXY /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)

    def calc_marginals(self):
        self.pdfX = np.sum(self.pdfXY, axis=0)
        self.pdfY = np.sum(self.pdfXY, axis=1)

        self.pdfX = conditional_correlation.normalize_pdf_1D(self.pdfX, self.lx, self.ux, self.resolution_x)
        self.pdfY = conditional_correlation.normalize_pdf_1D(self.pdfY, self.ly, self.uy, self.resolution_y)


class D_KL_conditional_marginal:
    def __init__(self, pdf1, pdf2, lower_bound_x, upper_bound_x, lower_bound_y, upper_bound_y):
        self.lx = lower_bound_x
        self.ux = upper_bound_x
        self.ly = lower_bound_y
        self.uy = upper_bound_y
        self.resolution_y, self.resolution_x = np.shape(pdf1)
        self.pdfXY1 = self.normalize_pdf_2D(pdf1)
        self.pdfXY2 = self.normalize_pdf_2D(pdf2)

    def normalize_pdf_2D(self, pdf):
        return pdf * self.resolution_x * self.resolution_y / (self.ux - self.lx) /\
               (self.uy - self.ly) / np.sum(pdf)

    def calc_DKL(self):

        self.DKL = np.sum(self.pdfXY1 * np.log(self.pdfXY1 / self.pdfXY2))
        self.DKL /= self.resolution_x * self.resolution_y / (self.ux - self.lx) / (self.uy - self.ly)
