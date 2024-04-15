import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree


def generate_points(sim_range, nsamples, ndim, points_type, seed=1, plot=0, min_dist=0, nodes1d=None):
    '''
    Generate points using different methods
    '''

    out = []
    np.random.seed(seed)
    if(type(sim_range) is float or type(sim_range) is int):

        sim_ranges = np.zeros((2, ndim))
        sim_ranges[0, :] = -sim_range  # low bound
        sim_ranges[1, :] = sim_range  # up bound

        sim_range = sim_ranges

    if(points_type == "random"):

def read_diamond_data(file_path, features):
    # Read data from the CSV file
    df = pd.read_csv(diamonds.csv)
    
    # Select the five independent variables and the dependent variable ("price")
    df_selected = df["carat", "depth", "table", "length", "width" + ["price"]]
    
   # Convert the selected features to a numpy array
    x_input = df_selected["carat", "depth", "table", "length", "width"].values
    
    # Extract the target variable ("price")
    y_output = df_selected["price"].values
    
    return x_input, y_output

    elif points_type == "regular":

        npts = round(pow(nsamples, 1./ndim))
        xs1d = np.zeros((ndim, npts))
        for d in range(ndim):

            xs1d[d] = np.linspace(sim_range[0][d], sim_range[1][d], npts)
            dx = xs1d[d][1]-xs1d[d][0]
            print("dx[", d, "]= ", dx)

        if ndim == 1:

            x1 = xs1d[0]
            x_input = x1.reshape(-1, 1)
            out.append(x_input)
            out.append(x1)

        elif ndim == 2:

            x1, x2 = np.meshgrid(xs1d[0], xs1d[1])
            x_input = np.stack((x1.reshape(-1), x2.reshape(-1)), axis=-1)

            out.append(x_input)
            out.append(x1)
            out.append(x2)

        elif ndim == 3:

            x1, x2, x3 = np.meshgrid(xs1d[0], xs1d[1], xs1d[2])
            x_input = np.stack(
                (x1.reshape(-1), x2.reshape(-1), x3.reshape(-1)), axis=-1)

            out.append(x_input)
            out.append(x1)
            out.append(x2)
            out.append(x3)

        else:
            raise ValueError("regular grid in not implemented for ndim>3")

    elif(points_type == "randomgrid"):

        if nodes1d is None:
            raise ValueError(
                f'nodes1d is required for points_type = {points_type}')

        x_input = np.zeros((nsamples, ndim))
        np.random.seed(seed)
        x_input = np.random.choice(nodes1d, size=(nsamples, ndim))
        out.append(x_input)

    else:
        raise ValueError("unknown points type")

    if ndim == 2 and plot:

        plt.figure()
        plt.plot(out[0][:, 0], out[0][:, 1], '.')
        plt.show()

    return out


def average_distance(points):
    '''
    calculates average distance between given points
    '''
    xs = points
    tree = KDTree(xs)
    dd, _ = tree.query(xs, k=2)
    avd = dd[:, 1].mean()
    return avd
