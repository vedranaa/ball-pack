#%%

import torch
import numpy as np
from icosphere import icosphere  # pip install icosphere
import plotly.graph_objects as go
from plotly.colors import qualitative
from tqdm.auto import tqdm  # problems with container width
# from tqdm import tqdm

RNG = torch.Generator().manual_seed(13)

## HELPING FUNCTIONS
def minimal_distance(radii):
    '''Minimal non-overlap distance between spheres with given radii.'''
    min_d = radii.unsqueeze(1) + radii.unsqueeze(0)
    min_d.fill_diagonal_(0)
    return min_d

def pairwise_distance(p, q = None):
    '''Pairwise distance between point-clouds.'''
    if q is None:
        q = p
    d = (p.unsqueeze(-1) - q.unsqueeze(-2)).norm(dim=-3)
    return d

def overlap_penalty(d, min_d):
    '''Overlap between balls.'''
    return torch.relu(torch.triu(min_d - d,  diagonal=1)).pow(2).sum()

# def separation_penalty(d, radii, n):
#     '''Separation to n nearest neighbors.'''
#     n = min(n, len(radii) - 1)
#     if n < 1:
#         return torch.tensor(0)
#     vals, inds = torch.topk(d, n + 1, largest=False)
#     vals, inds = vals[:, 1:], inds[:, 1:]
#     return torch.relu(vals - radii.unsqueeze(1) - radii[inds]).pow(2).sum()

def protrusion_penalty(xy, radii, R):
    '''Protrusion of balls outside the domain (radially).'''
    r = xy.norm(dim=-2)
    return torch.relu(r + radii - R).pow(2).sum()

def overhang_penalty(z, radii, Z):
    '''Overhang of balls outside the domain (transversely).'''
    return torch.relu(torch.abs(z - Z/2) + radii - Z/2).pow(2).sum()

def animation_controls(Z, prefix='Slice '):
    '''Animation controls for Z slices.'''
    layout = go.Layout(
        updatemenus=[{"buttons": [
            {"args": [None, {"frame": {"duration": 500, "redraw": True}, 
                "fromcurrent": True}], "label": ">", "method": "animate"},
            {"args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                "mode": "immediate", "transition": {"duration": 0}}],
                "label": "||", "method": "animate"}],
            "direction": "left", "pad": {"r": 10, "t": 30}, "showactive": False,
            "type": "buttons", "x": 0, "xanchor": "left", "y": 0, "yanchor": "top"}],
        sliders=[{"steps": [
            {"args": [[str(z)], { "frame": {"duration": 300, "redraw": True},
                "mode": "immediate", "transition": {"duration": 300}}],
                "label": str(z), "method": "animate"} for z in range(Z)],
            "x": 0.2, "len": 0.8, "xanchor": "left", "y": 0, "yanchor": "top",
            "pad": {"b": 10, "t": 10},
            "currentvalue": {"font": {"size": 15}, "prefix": prefix,
                "visible": True, "xanchor": "right"},
            "transition": {"duration": 300, "easing": "cubic-in-out"}}])
    return layout

def animate(volume, prefix='Slice ', figsize=600):
    '''Animate slices of a given volume.'''
    if type(volume) is torch.Tensor:
        volume = volume.numpy()
    vmin = volume.min()
    vrange = volume.max() - vmin
    nr_slices = volume.shape[0]
    frames = []
    for i in range(nr_slices):
        s = volume[i]
        s = (255 * (s - vmin) / vrange).astype(np.uint8)
        image = np.stack([s, s, s], axis=-1)  # plotly wants rgb image
        frames.append(go.Frame(data=[go.Image(z=image)], name=str(i)))
    layout = animation_controls(nr_slices, prefix=prefix)
    layout.update({'width': figsize, 'height': figsize})
    fig = go.Figure(layout=layout, frames=frames)
    fig.add_trace(frames[0].data[0])
    fig.show()

def t2s(tensor):
    '''Tensor to string, for exporting mesh.'''
    return ' '.join([str(x.item()) for x in tensor])

def save_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for i in range(vertices.shape[0]):
            f.write(f'v {t2s(vertices[i])}\n')
        for i in range(faces.shape[0]):
            f.write(f'f {t2s(faces[i])} \n')

def get_mapping_function(transition=None, r=0, sigma=None, edge=None):
    '''For mapping the signed distance field to intensity.'''

    # Handle transition parameter first
    if transition == 'smooth':
        sigma = 0.2 * r
    elif transition == 'enhanced':
        sigma = 0.2 * r
        edge = 0.5
    elif isinstance(transition, tuple):
        sigma, edge = transition
    
    # Return the mapping function
    if sigma is None:
        return lambda x: 1 - torch.heaviside(x, torch.tensor(0.5))
    
    def smooth_mapping(x):
        hg = 1 - 0.5 * (1 + torch.special.erf(x / (2**0.5 * sigma)))
        if edge is None:
            return hg
        return hg - edge * x / sigma * torch.exp(0.5 * (1 - (x / sigma)**2))
    
    return smooth_mapping



## BALL PACKER CLASS

def from_n(R, Z, N, r_mean, r_sigma=0, rng=None):
    bal = BallPacker()
    bal.R = R
    bal.Z = Z  
    bal.N = N
    if rng is not None:
        bal.rng = rng
    if r_sigma:
        radii = r_mean + r_sigma * torch.randn(N, generator=bal.rng)
        radii = torch.clamp(radii, 0.01 * r_mean)
    else:
        radii = r_mean * torch.ones(N)
    bal.set_radii(radii)
    bal.set_weights()
    bal.initialize()
    return bal

def from_vf(R, Z, vf, r_mean, r_sigma=0, rng=None):
    N = int(vf / 100 * (R**2) * Z / ((4/3) * r_mean**3)) # estimate number of balls  
    bal = from_n(R, Z, N, r_mean, r_sigma, rng)
    fix = (vf * (3/400) * (R**2) * Z / bal.radii.pow(3).sum()).pow(1/3)
    bal.set_radii(bal.radii * fix)
    return bal

class BallPacker():
    def __init__(self):
        self.R = None
        self.Z = None
        self.N = None
        self.radii = None
        
        self.configuration = None
    
        self.device = None
        self.rng = RNG
        self.learning_rate = 0.05
        
        self.weights = {
                'overlap': None, 
                # 'separation': None, 
                'protrusion': None,
                'overhang': None,
        }
        self.separation_neighbors = 3

        self.slicing = 50  # number of slices shown when animating 
        
        self.colors = qualitative.Plotly
        self.figsize = 600  # in pixels
        self.tqdmformat = '{desc:<50}{bar}{n_fmt}/{total_fmt}'
        self.tqdmwidth = 600  # in pixels for tqdm.notebook, else in characters
        
    ### INITIALIZATION METHODS

    def set_weights(self):
        self.weights = {
            'overlap': 1, 
            # 'separation': 1/1000,
            'protrusion': 100,
            'overhang': 100,
            }

    def set_radii(self, radii):
        self.radii = radii
        self.N = len(radii)
        self.min_d = minimal_distance(radii)

    def initialize(self):
        mean_r = self.radii.mean()
        max_R = self.R - mean_r
        ri = max_R * torch.sqrt(torch.rand(self.N, generator=self.rng))
        ai = torch.rand(self.N, generator=self.rng) * 2 * torch.pi
        zi = torch.rand(self.N, generator=self.rng) * (self.Z - 2 * mean_r) + mean_r
        self.configuration = torch.stack((ri * torch.cos(ai), ri * torch.sin(ai), zi))

    ### VISUALIZATION METHODS

    def show_radii_distribution(self, nbins=100):
        # I don't know exactly how plotly computes number of bins from nbinsx
        data = [go.Histogram(x=self.radii, nbinsx=nbins)]
        fig = go.Figure(data=data)
        vf = (400/3) * self.radii.pow(3).sum() / (self.R**2 * self.Z) 
        title = f"Radii distribution, N = {self.N}, mean = {self.radii.mean():.2f}, vf = {vf:.1f}"
        fig.update_layout(title=title, xaxis_title='Radii', yaxis_title="Count", 
                width=self.figsize, height=self.figsize//2)
        fig.show()
    
    def get_slice_circles(self, s):
        '''A helping function for show_slice, slices balls into circles.'''
        x, y, z = self.configuration
        d = torch.abs(z - s)
        shapes = []
        # animation only works if all balls are present in all slices so 
        # I set radius=0 for balls not visible in slice
        for i, r in enumerate(self.radii):
            sr = torch.sqrt(r**2 - d[i]**2) if d[i] < r else 0
            shapes.append(dict(x0=x[i] - sr, y0=y[i] - sr, 
                        x1=x[i] + sr, y1=y[i] + sr, type="circle", 
                        fillcolor=self.color(i), opacity=0.5, line_width=0))
        shapes.append(dict(x0=-self.R, y0=-self.R, x1=self.R, y1=self.R,
                type="circle", line_color='gray', opacity=0.5, line_width=5))
        return shapes

    def show_slice(self, s, title=None):
        fig = go.Figure()
        shapes = self.get_slice_circles(s)
        fig.update_layout(shapes=shapes)
        fig.update_layout(self.get_layout())
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    def color(self, i):
        '''Color for i-th ball.'''
        return self.colors[i % len(self.colors)]

    def show_3D_configuration(self, title=None, scale=5):
        x, y, z = self.configuration
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', 
                marker_color = [self.color(i) for i in range(self.N)], 
                marker=dict(size=scale * self.radii)))
        fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y',
            zaxis_title='Z', aspectmode='cube'), showlegend=False, 
            width=self.figsize, height=self.figsize)
        if title:
            fig.update_layout(title=title)
        fig.show()

    def get_layout(self):
        layout = {'xaxis': dict(range=[-self.R, self.R]),
                'yaxis': dict(range=[-self.R, self.R], scaleanchor="x"),
                'width': self.figsize, 'height': self.figsize}
        return layout

    def animate_slices(self, title=None):
        frames = []
        for i, s in enumerate(torch.linspace(0, self.Z, self.slicing)):
            shapes = self.get_slice_circles(s)
            frames.append(go.Frame(layout=dict(shapes=shapes), name=str(i)))
        layout = animation_controls(self.slicing)
        layout.update(self.get_layout())   
        fig = go.Figure(layout=layout, frames=frames)
        fig.update_layout(shapes=frames[0].layout.shapes)
        if title:
            fig.update_layout(title=title)
        fig.show()
    
    ### OPTIMIZATION METHODS

    def select_device(self):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        print(f"Using device {self.device}")
        return self.device

    def optimize_configuration(self, iters=500):
        
        if self.device is None:
            self.select_device()

        configuration = self.configuration.to(self.device)
        configuration.requires_grad = True
        min_d = self.min_d.to(self.device)
        radii = self.radii.to(self.device)
        weights = self.weights

        optimizer = torch.optim.Adam([configuration], lr=self.learning_rate)
        losses = {key: [] for key in weights.keys()}
        progress_bar = tqdm(range(iters), bar_format=self.tqdmformat, 
                ncols=self.tqdmwidth)
        for iter in progress_bar:  
            optimizer.zero_grad()   
            d = pairwise_distance(configuration)
            overlap = overlap_penalty(d, min_d)
            overlap = overlap * weights['overlap']
            protrusion = protrusion_penalty(configuration[:2], radii, self.R)
            protrusion = protrusion * weights['protrusion']
            overhang = overhang_penalty(configuration[2], radii, self.Z) 
            overhang = overhang * weights['overhang']

            loss = overlap + protrusion + overhang
            loss.backward()
            optimizer.step()

            losses['overlap'].append(overlap.item())
            losses['protrusion'].append(protrusion.item())
            losses['overhang'].append(overhang.item())

            progress_bar.set_description(
                f"Overlap {overlap:.2f}, protrusion. {protrusion:.1f}, "
                f"overhang. {overhang:.1f}",
                refresh=True
            )

        self.configuration = configuration.detach().to('cpu')
        return losses


    def show_losses(self, losses):
        fig = go.Figure()
        for k, v in losses.items():
            fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, 
                    mode='lines', name=k))
        fig.update_layout(title='Loss contributions (log scale)', 
                xaxis_title='Iteration', yaxis_title='Loss', yaxis_type='log', 
                width=self.figsize, height=self.figsize//2) 
        fig.show()

    
    def save_mesh(self, filename, nu=5, center=False):

        v, f = icosphere(nu)
        v, f = torch.tensor(v), torch.tensor(f)
        l = v.shape[0]
        xyz = self.configuration.clone()
        if center:
            xyz[2] = xyz[2] - self.Z/2 
        else:
            xyz[:2] = xyz[:2] + self.R 
        faces = []
        vertices = []
        for i in range(self.N):
            vertices.append(self.radii[i] * v + xyz[:, i])
            faces.append(f + i * l)
        
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        save_obj(filename, vertices, faces + 1)
        print(f"Saved to {filename}")


    def voxelize(self, transition=None):
        '''Assumes that self.Z and self.R are in pixel units'''
        r = int(np.ceil(self.R))
        xy = torch.arange(-r, r + 1, 1)
        zz = torch.arange(0, int(np.ceil(self.Z)) + 1, 1)
        Z, Y, X = torch.meshgrid(zz, xy, xy, indexing='ij')
        df = torch.full((len(zz), len(xy), len(xy)), float(self.R))
        x, y, z = self.configuration
        for i in range(self.N):
            dist = ((X - x[i])**2 + (Y - y[i])**2 + (Z - z[i])**2).sqrt() - self.radii[i]
            df = torch.minimum(df, dist)
        mapping_function = get_mapping_function(transition, self.radii.mean())
        df = mapping_function(df)
        return df


# %%
