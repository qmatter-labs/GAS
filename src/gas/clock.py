import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MPath

### Code for plotting a binary clock
### used in paper to generate figures


def clock_to_math(theta):
    """Convert clock angle (0 at top, clockwise) to math angle (0 at right, CCW)."""
    return np.pi / 2 - theta


def create_binary_clock(n, signed=False, figsize=(10, 10), title=False):
    """
    Create a binary clock with n bits.
    
    Parameters:
    n : int - number of bits (creates a polygon with 2^n vertices)
    signed : bool - if True, use signed range -2^(n-1) to 2^(n-1)-1
    """
    # Number of positions
    num_positions = 2**n
    
    # Generate values in clockwise order starting from top (0)
    if signed:
        # Signed two's complement range
        max_pos = 2**(n-1) - 1
        min_neg = -2**(n-1)
        # Order: 0, positive numbers clockwise, then negative numbers clockwise
        positives = list(range(1, max_pos + 1))
        negatives = list(range(min_neg, 0))
        values = [0] + positives + negatives
    else:
        values = list(range(num_positions))
    
    # Clock angles (0 at top, clockwise)
    clock_angles = np.linspace(0, 2*np.pi, num_positions, endpoint=False)
    
    # Convert to math angles for plotting
    math_angles = clock_to_math(clock_angles)
    
    # Get vertex coordinates
    x = np.cos(math_angles)
    y = np.sin(math_angles)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # -------- DRAW POLYGON --------
    # Close the polygon
    x_poly = np.append(x, x[0])
    y_poly = np.append(y, y[0])
    
    # Draw polygon with fill
    ax.fill(x_poly, y_poly, alpha=0.1, color='skyblue', zorder=1)
    ax.plot(x_poly, y_poly, 'b-', linewidth=2, zorder=2)
    
    # -------- DRAW VERTICES AS CIRCLES --------
    ax.scatter(x, y, s=100, c='white', edgecolors='navy', linewidth=2, zorder=3)
    
    # -------- ADD LABELS --------
    for i, (xi, yi) in enumerate(zip(x, y)):
        val = values[i]
        
        # Binary representation
        if signed and val < 0:
            binary = format(2**n + val, f'0{n}b')
        else:
            binary = format(val, f'0{n}b')
        
        # Decimal label (slightly outside the polygon)
        radius_dec = 1.1
        ax.text(radius_dec * xi, radius_dec * yi, str(val),
                ha='center', va='center',
                # fontsize=12,
                fontweight='bold', color='darkblue')
        
        # Binary label (just inside the polygon)
        radius_bin = 0.85
        ax.text(radius_bin * xi, radius_bin * yi, binary,
                ha='center', va='center',
                # fontsize=10, 
                family='monospace', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', edgecolor='orange', alpha=0.9))
    
    # -------- CENTER DOT --------
    ax.plot(0, 0, 'o', color='navy', markersize=12, zorder=4)
    ax.plot(0, 0, 'o', color='white', markersize=8, zorder=5)
    
    # -------- TITLE --------
    if title:
        if signed:
            title = f'{n}-Bit Binary Clock (Signed: {min_neg} to {max_pos})'
        else:
            title = f'{n}-Bit Binary Clock (0 to {2**n-1})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    return fig, ax, clock_angles, values, x, y


def add_hand(ax, clock_angle, length=0.8, color='red', width=3, 
             alpha=0.8, label=None):
    """Add a hand to the clock at the specified clock angle."""
    math_angle = clock_to_math(clock_angle)
    
    arrow = FancyArrowPatch((0, 0), 
                           (length * np.cos(math_angle), length * np.sin(math_angle)),
                           arrowstyle='-|>', mutation_scale=20,
                           color=color, linewidth=width, alpha=alpha)
    ax.add_patch(arrow)
    
    if label:
        label_radius = length + 0.1
        ax.text(label_radius * np.cos(math_angle), 
                label_radius * np.sin(math_angle),
                label, ha='center', va='center',
                fontsize=10, color=color, fontweight='bold')


def add_movement_arc(ax, start_clock, end_clock, radius=0.7, 
                      color='purple', linewidth=2.5, 
                      clockwise=True, arrow_size=12, alpha=0.8):
    """
    Add an arc showing movement from start_clock to end_clock.
    The arc stays exactly on the circular path at the given radius.
    
    Parameters:
    start_clock, end_clock : float - clock angles in radians (0 at top, clockwise)
    radius : float - radius of the arc
    clockwise : bool - if True, arc goes clockwise from start to end
                      if False, arc goes anticlockwise from start to end
    arrow_size : int - mutation scale for the arrowhead
    """
    # Convert to math angles
    start_math = clock_to_math(start_clock)
    end_math = clock_to_math(end_clock)
    
    if clockwise:
        # Clockwise: go in the direction of increasing clock angle
        # In math coordinates, this means decreasing angle
        # Calculate the clockwise distance from start to end
        dist = (start_math - end_math) % (2*np.pi)
        if dist == 0:
            dist = 2*np.pi  # Full circle
            
        # Generate points along the clockwise arc
        # Use many points for a smooth arc
        t = np.linspace(0, dist, 100)
        arc_math = start_math - t
        
        # Handle wrap-around for angles that cross 0
        arc_math = np.where(arc_math < 0, arc_math + 2*np.pi, arc_math)
    else:
        # Anticlockwise: go opposite to clockwise direction
        # Calculate the anticlockwise distance from start to end
        dist = (end_math - start_math) % (2*np.pi)
        if dist == 0:
            dist = 2*np.pi  # Full circle
            
        # Generate points along the anticlockwise arc
        t = np.linspace(0, dist, 100)
        arc_math = start_math + t
        
        # Handle wrap-around for angles that exceed 2π
        arc_math = np.where(arc_math > 2*np.pi, arc_math - 2*np.pi, arc_math)
    
    # Ensure the arc is monotonically changing in the right direction
    # This prevents any backtracking that could cause the arrow to go off-circle
    
    # Coordinates for the arc - exactly on the circle at the given radius
    x_arc = radius * np.cos(arc_math)
    y_arc = radius * np.sin(arc_math)
    
    # Create vertices for the path
    vertices = np.column_stack([x_arc, y_arc])
    
    # Create a simple path with LINETO commands (no curves that might deviate)
    codes = [MPath.MOVETO] + [MPath.LINETO] * (len(vertices) - 1)
    
    # Create the path
    path = MPath(vertices, codes)
    
    # Create a FancyArrowPatch along the path
    arrow = FancyArrowPatch(path=path,
                           arrowstyle='->', 
                           mutation_scale=arrow_size,
                           color=color, 
                           linewidth=linewidth, 
                           alpha=alpha,
                           shrinkA=0,
                           shrinkB=0)
    ax.add_patch(arrow)
    
    # Optional: Mark start and end points for verification
    # ax.plot(x_arc[0], y_arc[0], 'o', color=color, markersize=4, alpha=0.5)
    # ax.plot(x_arc[-1], y_arc[-1], 's', color=color, markersize=4, alpha=0.5)


def add_simple_arc(ax, start_clock, end_clock, radius=0.7, 
                    color='purple', linewidth=2.5,
                    clockwise=True, arrow_size=10):
    """
    Simpler implementation using plot for the arc and a small arrow at the end.
    This ensures the arc stays exactly on the circle.
    """
    # Convert to math angles
    start_math = clock_to_math(start_clock)
    end_math = clock_to_math(end_clock)
    
    if clockwise:
        # Clockwise distance
        dist = (start_math - end_math) % (2*np.pi)
        if dist == 0:
            dist = 2*np.pi
        t = np.linspace(0, dist, 200)
        arc_math = start_math - t
        arc_math = np.where(arc_math < 0, arc_math + 2*np.pi, arc_math)
    else:
        # Anticlockwise distance
        dist = (end_math - start_math) % (2*np.pi)
        if dist == 0:
            dist = 2*np.pi
        t = np.linspace(0, dist, 200)
        arc_math = start_math + t
        arc_math = np.where(arc_math > 2*np.pi, arc_math - 2*np.pi, arc_math)
    
    # Coordinates exactly on the circle
    x_arc = radius * np.cos(arc_math)
    y_arc = radius * np.sin(arc_math)
    
    # Draw the main arc line
    ax.plot(x_arc, y_arc, color=color, linewidth=linewidth, alpha=0.8)
    
    # Add a small arrow at the end using the last two points
    if len(x_arc) > 1:
        # Use the last two points to determine direction
        last_x = x_arc[-2]
        last_y = y_arc[-2]
        end_x = x_arc[-1]
        end_y = y_arc[-1]
        
        # Calculate direction vector
        dx = end_x - last_x
        dy = end_y - last_y
        
        # Normalize and scale for arrow length
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Draw a small arrow at the end
            arrow = FancyArrowPatch((last_x, last_y), (end_x, end_y),
                                   arrowstyle='->', 
                                   mutation_scale=arrow_size,
                                   color=color, 
                                   linewidth=linewidth,
                                   alpha=0.8,
                                   shrinkA=0,
                                   shrinkB=0)
            ax.add_patch(arrow)

