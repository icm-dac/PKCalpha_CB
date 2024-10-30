import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, morphology, filters, util
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import binary_closing, disk, skeletonize, remove_small_objects, thin, medial_axis
from skimage.filters import gaussian, threshold_otsu, apply_hysteresis_threshold
from scipy.cluster.hierarchy import fclusterdata
import networkx as nx
import sknw
from shapely.geometry import LineString
import os
from skimage import exposure
import random
from scipy.spatial import distance_matrix
from scipy.ndimage import uniform_filter1d


# Ensure file extension is aligned with expectation
def is_image_file(filename):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


# Function to check if the image is almost blank
# to detect non-transfected cells automatically
def is_blank(image, threshold=0.05):
    # Calculate the percentage of non-zero pixels in the image
    non_zero_pixels = np.count_nonzero(image)
    total_pixels = image.size
    return (non_zero_pixels / total_pixels) < threshold

# Directories containing images
dir_path = '/path_1'
alexa_dir_path = '/path_2'
csv_output_dir = '/path_3'
user_csv_filename = 'junctional_intensity'
images_output_dir = '/path_4' 


# Create the CSV output directory if it does not exist
if not os.path.exists(csv_output_dir):
    os.makedirs(csv_output_dir)

# List all files in both directories
image_files = sorted([f for f in os.listdir(dir_path) if is_image_file(f)])
alexa_image_files = sorted([f for f in os.listdir(alexa_dir_path) if is_image_file(f)])

# Ensure both folders have the same number of files
assert len(image_files) == len(alexa_image_files), "Folders contain a different number of files"


class JunctionsAnalyzer:
    def __init__(self, image):
        self.image = image
        self.graph = nx.Graph()

    def preprocess_image(self):
        if self.image.ndim == 3:
            # Convert to grayscale if it's a color image
            self.image = color.rgb2gray(self.image)
        thresh = threshold_otsu(self.image)
        binary = self.image > thresh
        return binary.astype(np.uint8) 

    def skeletonize(self):
        self.skeleton = self.thin_and_skeletonize(self.binary)

    def thin_and_skeletonize(self, mask):
        if mask.ndim != 2:
            raise ValueError("Mask must be a 2-dimensional array")
        _thin_mask = thin(mask)
        return skeletonize(_thin_mask > 0)

    def build_graph(self):
        self.graph = sknw.build_sknw(self.skeleton)

    def detect_nodes_edges(self, binary):
        self.binary = binary
        self.skeletonize()
        self.build_graph()
        nodes, edges = self.extract_nodes_and_edges_from_graph()
        return nodes, edges

    def extract_nodes_and_edges_from_graph(self):
        nodes = [(data['o'][1], data['o'][0]) for node, data in self.graph.nodes(data=True)]
        edges = []
        for (s, e) in self.graph.edges():
            edges.append(self.graph[s][e]['pts'][:, [1, 0]].tolist())  # Reversing x and y
        return nodes, edges

    def extract_intersecting_parts(self, intersections):
        intersecting_nodes = []
        intersecting_edges = []

        for node in self.graph.nodes:
            x, y = self.graph.nodes[node]['o']
            if any(abs(x - ix) <= 5 and abs(y - iy) <= 5 for ix, iy in intersections):
                intersecting_nodes.append((x, y))

        for edge in self.graph.edges:
            pts = self.graph[edge[0]][edge[1]]['pts']
            if any(any(abs(x - ix) <= 5 and abs(y - iy) <= 5 for x, y in pts) for ix, iy in intersections):
                intersecting_edges.append(pts)

        return intersecting_nodes, intersecting_edges

    def visualize_nodes_edges(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image, cmap='gray')

        # Draw edges
        for (s, e) in self.graph.edges():
            pts = self.graph[s][e]['pts']
            plt.plot(pts[:,1], pts[:,0], 'blue')

        # Draw nodes
        for node, data in self.graph.nodes(data=True):
            pts = data['o']
            plt.plot(pts[1], pts[0], 'ro')

        plt.title('Nodes and Edges Visualization')
        plt.axis('off')
        plt.show()

    def visualize_nodes_edges_with_labels(self, overlay_image=None, specific_edges=None):
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 10))

        # Check if an overlay image is provided
        if overlay_image is not None:
            # Ensure overlay_image is in color
            if len(overlay_image.shape) == 2 or (len(overlay_image.shape) == 3 and overlay_image.shape[2] == 1):
                overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)  # Make sure this conversion is appropriate for your image data

            # Display the overlay image
            ax.imshow(overlay_image, cmap='gray', alpha=0.5)
        else:
            # If no overlay, just show the current image
            ax.imshow(self.image, cmap='gray')

        # Decide whether to visualize all edges or specific ones
        edges_to_draw = specific_edges if specific_edges is not None else self.graph.edges()

        # Draw edges with labels
        for i, (s, e) in enumerate(edges_to_draw):
            pts = self.graph[s][e]['pts']
            ax.plot(pts[:, 1], pts[:, 0], 'blue')
            label_pos = (np.mean(pts[:, 1]), np.mean(pts[:, 0]))
            # Change text color to 'yellow' or another high-contrast color, and increased font size for visibility
            ax.text(label_pos[0], label_pos[1], str(i), color='green', fontsize=14, ha='center', va='center', zorder=3)

        # Draw nodes
        for node, data in self.graph.nodes(data=True):
            pts = data['o']
            # Increased node point size for visibility
            ax.plot(pts[1], pts[0], 'ro', markersize=5)

        # Set limits and title
        ax.set_xlim([0, self.image.shape[1]])
        ax.set_ylim([self.image.shape[0], 0])
        plt.title('Nodes and Edges Visualization with Labels')
        plt.axis('off')
        plt.show()


def preprocess_alexa_for_ctnnd1(image):
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define the range of colors to keep
    lower_bound = np.array([0, 0, 65])  # Hmin, Smin, Vmin
    upper_bound = np.array([179, 255, 255])  # Hmax, Smax, Vmax
    
    # Create a mask that includes colors within the specified range
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    
    # Apply the mask to get the filtered image
    filtered_image = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
    
    # Convert back to RGB for further processing
    return cv2.cvtColor(filtered_image, cv2.COLOR_HSV2RGB)


def enhance_honeycomb(image, sigma=1):
    image_blur = gaussian(image, sigma=sigma)
    selem = disk(3)
    image_closed = morphology.closing(image_blur, selem)
    high_thresh = threshold_otsu(image_closed)
    low_thresh = high_thresh * 0.9
    binary_mask = apply_hysteresis_threshold(image_closed, low_thresh, high_thresh)
    binary_mask_closed = binary_closing(binary_mask, disk(3))
    return binary_mask_closed

def mark_intersections(image, pairs, skeleton, binary_region, radius=5):
    intersections = []  # List to store intersection points
    for (cX1, cY1), (cX2, cY2) in pairs:
        # Calculate midpoint
        midX, midY = (cX1 + cX2) // 2, (cY1 + cY2) // 2
        
        # Check if the midpoint intersects with the skeleton
        if skeleton[midY, midX]:
            intersections.append((midX, midY))
            cv2.circle(image, (midX, midY), radius, (0, 255, 0), -1)
        
        # Draw the parallel line through midpoint
        draw_parallel_line_through_midpoint(image, (cX1, cY1), (cX2, cY2), binary_region)
    
    return intersections

def draw_parallel_line_through_midpoint(image, p1, p2, binary_region, color=(0, 0, 255), thickness=2):
    # Calculate midpoint
    midX, midY = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
    
    # Calculate direction vector for the line between centroids
    dir_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    
    # Calculate a normalized perpendicular vector
    perp_vec = np.array([-dir_vec[1], dir_vec[0]])
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    
    # Define a length for how far the line should extend
    length = 100  # Adjust as needed
    
    # Calculate start and end points of the parallel line
    start_point = (int(midX - perp_vec[0] * length), int(midY - perp_vec[1] * length))
    end_point = (int(midX + perp_vec[0] * length), int(midY + perp_vec[1] * length))
    
    # Clip the line to the image boundaries
    start_point = (np.clip(start_point[0], 0, binary_region.shape[1]-1), np.clip(start_point[1], 0, binary_region.shape[0]-1))
    end_point = (np.clip(end_point[0], 0, binary_region.shape[1]-1), np.clip(end_point[1], 0, binary_region.shape[0]-1))
    
    # Draw the parallel line
    cv2.line(image, start_point, end_point, color, thickness)

# Function to calculate centroids from contours
def calculate_centroids(contours):
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))
    return centroids


def find_nearest_pairs(centers):
    centroids_pairs = []
    for i, center1 in enumerate(centers):
        min_dist = float('inf')
        nearest_center = None
        for j, center2 in enumerate(centers):
            if i != j:
                dist = np.linalg.norm(np.array(center1) - np.array(center2))
                if dist < min_dist:
                    min_dist = dist
                    nearest_center = (center1, center2)
        if nearest_center:
            centroids_pairs.append(nearest_center)
    return centroids_pairs



def mark_intersections(image, pairs, skeleton, search_radius=5, radius=5):
    # First, normalize the input if it is an ndarray regardless of its size
    if isinstance(search_radius, np.ndarray):
        if search_radius.size >= 1:
            search_radius = search_radius.flatten()[0]  # Take the first element safely
        else:
            raise ValueError("search_radius is an empty array, which is invalid.")
    search_radius = int(search_radius)  # Convert to integer

    if isinstance(radius, np.ndarray):
        if radius.size >= 1:
            radius = radius.flatten()[0]  # Take the first element safely
        else:
            raise ValueError("radius is an empty array, which is invalid.")
    radius = int(radius)  # Convert to integer

    for (cX1, cY1), (cX2, cY2) in pairs:
        # Draw line between centroids
        cv2.line(image, (cX1, cY1), (cX2, cY2), (255, 0, 0), 2)
        
        # Calculate midpoint
        midX, midY = (cX1 + cX2) // 2, (cY1 + cY2) // 2
        
        # Search for the closest skeleton point to the midpoint within the search radius
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                newY, newX = midY + dy, midX + dx
                if newY >= 0 and newY < skeleton.shape[0] and newX >= 0 and newX < skeleton.shape[1]:
                    if skeleton[newY, newX]:
                        cv2.circle(image, (newX, newY), radius, (0, 255, 0), -1)
                        break
            else:
                continue
            break



def analyze_intersecting_edges(edges, centroid_lines, foreground_image):
    # This will store the results
    edge_analysis = []

    for line in centroid_lines:
        for edge in edges:
            if check_intersection(line, edge):
                # Calculate the properties of the intersecting edge
                edge_centroid = calculate_edge_centroid(edge)
                edge_length = calculate_edge_length(edge)
                edge_intensity = calculate_edge_intensity(edge, foreground_image)

                # Store the results
                edge_analysis.append({
                    'centroid': edge_centroid,
                    'length': edge_length,
                    'average_intensity': edge_intensity
                })

    return edge_analysis


def save_intersecting_edges_to_csv(intersecting_edges_info, image_path, csv_file_path):
    filename, protein_name, condition_name, experiment_no, image_no = parse_filename(image_path)
    updated_edges_info = []

    for edge_info in intersecting_edges_info:
        centroid_x, centroid_y = edge_info['centroid']
        updated_edge_info = {
            'image_name': filename,
            'protein_name': protein_name,
            'condition_name': condition_name,
            'experiment_no': experiment_no,
            'image_no': image_no,
            'label': edge_info['label'],
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'length': edge_info['length'],
            'average_intensity': edge_info['average_intensity']
        }
        updated_edges_info.append(updated_edge_info)

    df = pd.DataFrame(updated_edges_info)
    columns_order = ['image_name', 'protein_name', 'condition_name', 'experiment_no', 'image_no', 'label', 'centroid_x', 'centroid_y', 'length', 'average_intensity']
    df = df[columns_order]
    df.to_csv(csv_file_path, index=False)


def prepare_edges_info_for_csv(intersecting_edges_info, normalization_value, image_path):
    filename, protein_name, condition_name, experiment_no, image_no = parse_filename(image_path)
    updated_edges_info = []

    for edge_info in intersecting_edges_info:
        centroid_x, centroid_y = edge_info.get('centroid', (0, 0))  # Use get with default
        normalized_intensity = edge_info.get('average_intensity', 0) / normalization_value if normalization_value else 0

        updated_edge_info = {
            'image_name': filename,
            'protein_name': protein_name,
            'condition_name': condition_name,
            'experiment_no': experiment_no,
            'image_no': image_no,
            'label': edge_info.get('label', 'default_label'),  # Default label if not found
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'length': edge_info.get('length', 0),
            'average_intensity': edge_info.get('average_intensity', 0),
            'normalize_value': normalization_value,
            'normalized_intensity': normalized_intensity
        }
        updated_edges_info.append(updated_edge_info)
    
    return updated_edges_info




def check_intersection(centroid_line, edge_points):
    # Convert the line and edge into Shapely LineStrings
    line = LineString(centroid_line)
    edge = LineString(edge_points)
    
    # Check if the line and edge intersect
    return line.intersects(edge)

def calculate_edge_centroid(edge_points):
    try:
        line = LineString(edge_points)
        return (line.centroid.x, line.centroid.y)
    except Exception as e:
        print(f"Error calculating centroid: {e}")
        return (0, 0)  # Default or error value

def calculate_edge_length(edge_points):
    # Use Shapely to calculate the length of the line
    edge = LineString(edge_points)
    return edge.length

def calculate_edge_intensity(edge_points, image):
    # Sample the pixel values from the image along the edge and calculate their average intensity
    intensities = [image[int(pt[1]), int(pt[0])] for pt in edge_points if 0 <= int(pt[1]) < image.shape[0] and 0 <= int(pt[0]) < image.shape[1]]
    return np.mean(intensities) if intensities else 0


def calculate_and_print_edge_properties(graph, image, image_path):
    filename, protein_name, condition_name, experiment_no, image_no = parse_filename(image_path)
    edge_properties = []
    
    for i, (s, e) in enumerate(graph.edges()):
        pts = np.array(graph[s][e]['pts'])
        edge_line = LineString(pts)
        centroid = edge_line.centroid.coords[0]  # get centroid coords
        length = edge_line.length
        intensities = [image[int(pt[0]), int(pt[1])] for pt in pts if 0 <= int(pt[0]) < image.shape[0] and 0 <= int(pt[1]) < image.shape[1]]
        average_intensity = np.mean(intensities) if intensities else 0
        
        edge_properties.append({
            'image_name': filename,
            'protein_name': protein_name,
            'condition_name': condition_name,
            'experiment_no': experiment_no,
            'label': i,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'length': length,
            'average_intensity': average_intensity
        })
    
    # Print edge properties
    for edge_prop in edge_properties:
        print(f"Edge {edge_prop['label']}: Centroid: {edge_prop['centroid_x'], edge_prop['centroid_y']}, Length: {edge_prop['length']}, Average Intensity: {edge_prop['average_intensity']}")
    # Save to CSV
    df = pd.DataFrame(edge_properties)
    df.to_csv('edges_info.csv', index=False)



def parse_filename(image_path):
    filename = os.path.basename(image_path)
    print("[PROCESSING] Filename:", filename)  # Debugging output

    # Split the filename using '-' to separate major components
    primary_split = filename.split(' - ')
    if len(primary_split) < 2:
        raise ValueError("Filename does not properly separate parts with ' - '.")

    # Handle the first part for chemical, condition, and experiment numbers
    details_part = primary_split[0].split()
    if len(details_part) < 3:
        raise ValueError("Filename does not contain enough details for chemical name, condition name, and experiment no.")

    protein_name = details_part[0]
    condition_name = details_part[1]
    experiment_no = details_part[2]

    # Optionally handle other parts like 'image_no' if needed from the second part of primary_split
    image_no = primary_split[1].split('_')[0]

    return filename, protein_name, condition_name, experiment_no, image_no



def connect_centroids_within_threshold(image, centers, max_distance=100, max_connections=10):
    if not centers or len(centers) < 2:
        print("Insufficient centers for connection.")
        return []  # Or appropriate handling

    # Calculate pairwise distances between all centers
    dist_matrix = distance_matrix(centers, centers)
    
    # Create an empty list to hold lines to be drawn (connections)
    lines = []
    
    # Iterate through each center and find its connections based on the distance criteria
    for i, center1 in enumerate(centers):
        connections = []  # Track connections for the current center
        
        # Iterate over all potential connections
        for j, center2 in enumerate(centers):
            if i != j:  # Avoid self-connection
                dist = np.linalg.norm(np.array(center1) - np.array(center2))
                if dist < max_distance:
                    connections.append((dist, j))
        
        # Sort connections by distance and take the closest ones up to max_connections
        connections.sort()
        for _, j in connections[:max_connections]:
            lines.append((center1, centers[j]))
            cv2.line(image, center1, centers[j], (255, 255, 0), 2)  # Drawing the line

    return lines



def extract_edges_intersecting_with_lines(edges, centroid_lines, image):
    intersecting_edges = []
    intersecting_edges_info = []

    for i, edge in enumerate(edges):
        for line in centroid_lines:
            centroid_line = LineString([line[0], line[1]])
            edge_line = LineString(edge)
            if centroid_line.intersects(edge_line):
                intersecting_edges.append(edge)
                centroid = edge_line.centroid.coords[0]
                length = edge_line.length
                # Sample pixel values along the edge
                intensities = [image[int(pt[1]), int(pt[0])] for pt in edge if 0 <= int(pt[1]) < image.shape[0] and 0 <= int(pt[0]) < image.shape[1]]
                average_intensity = np.mean(intensities) if intensities else 0

                intersecting_edges_info.append({
                    'label': i,
                    'centroid': centroid,
                    'length': length,
                    'average_intensity': average_intensity
                })
                break  # Stop checking other lines once an intersection is found
    
    return intersecting_edges, intersecting_edges_info


def connect_closest_centroids(centers, max_connections=3, max_distance=np.inf):
    # Compute the pairwise distance matrix
    dist_matrix = distance_matrix(centers, centers)

    # Set the distances of the same point to infinity to avoid self-connection
    np.fill_diagonal(dist_matrix, np.inf)

    # Create a list to hold the connections
    connections = []

    # Iterate through each center
    for idx, center in enumerate(centers):
        # Find the distances to all other points
        dists = dist_matrix[idx]

        # Filter out distances beyond the max distance
        close_points = [i for i, d in enumerate(dists) if d < max_distance]

        # Sort points by distance
        sorted_points = sorted(close_points, key=lambda i: dists[i])

        # Connect to the closest points, up to the maximum connections allowed
        for point_idx in sorted_points[:max_connections]:
            if point_idx != idx:  # Avoid connecting a center to itself
                connections.append((center, centers[point_idx]))

    return connections


def convert_points_to_edge_tuples(points_list, graph):
    edge_tuples = []
    for points in points_list:
        # Assuming 'points' is a NumPy array of points along an edge
        # You'd need to find which edge in the graph these points correspond to
        for (s, e) in graph.edges():
            if np.array_equal(graph[s][e]['pts'], points):
                edge_tuples.append((s, e))
                break
    return edge_tuples


def create_overlay_image(base_image, lines, color=(255, 255, 0), thickness=2):
    # Start with a copy of the base image
    overlay_image = base_image.copy()

    # Ensure base_image is in color
    if len(base_image.shape) == 2 or base_image.shape[2] == 1:
        overlay_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2RGB)

    # Draw each line on the overlay image
    for line in lines:
        cv2.line(overlay_image, line[0], line[1], color, thickness)
    
    return overlay_image


def apply_inverted_mask(image, mask):
    # Ensure the mask is in the correct format to apply it as transparency
    transparent_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    transparent_mask[:, :, 3] = mask  # Set alpha channel

    # Apply mask to the image
    return cv2.bitwise_and(image, image, mask=mask)


def calculate_average_intensity_of_random_edges(edges, image, num_edges):
    if len(edges) > num_edges:
        selected_edges = random.sample(edges, num_edges)
    else:
        selected_edges = edges

    intensities = [calculate_edge_intensity(edge, image) for edge in selected_edges]
    return np.mean(intensities) if intensities else 0


def calculate_moving_average_intensity(edges, image, window_size=10):
    intensities = [calculate_edge_intensity(edge, image) for edge in edges]
    if len(intensities) >= window_size:
        moving_average = uniform_filter1d(intensities, size=window_size)
        return np.mean(moving_average)
    else:
        return np.mean(intensities) if intensities else 0


def derive_edges_from_centers(centers):
    # Example: Create edges by connecting each center to its nearest neighbor
    if not centers:
        return []
    dist_mat = distance_matrix(centers, centers)
    np.fill_diagonal(dist_mat, np.inf)
    edges = []
    for i, center in enumerate(centers):
        nearest_index = np.argmin(dist_mat[i])
        if nearest_index != i:  # Prevent self-loop
            edges.append((center, centers[nearest_index]))
    return edges


def execute_main(edge_count):
    all_edges_info = []

    # Iterate through each file pair
    for image_file, alexa_image_file in zip(image_files, alexa_image_files):
        # Construct full file paths
        try:
            image_path = os.path.join(dir_path, image_file)
            image_path_alexa = os.path.join(alexa_dir_path, alexa_image_file)

            # Read and process images
            image = io.imread(image_path)
            image_alexa = io.imread(image_path_alexa)


            if image.ndim > 2 and image.shape[2] == 4:
                image = color.rgba2rgb(image)
            image_gray = color.rgb2gray(image)

            filename, protein_name, condition_name, experiment_no, image_no = parse_filename(image_path)

            # Check if the image is associated with CTNND1 and preprocess specifically for it
            if 'CTNND1' in protein_name:
                print(f"[PROCESSING] CTNND1 detected for {image_file}. Special preprocessing for {alexa_image_file}.")
                image_alexa = preprocess_alexa_for_ctnnd1(image_alexa)
         
            
            is_untransfected = 'untransfected' in condition_name.lower()
            
            if is_untransfected:
                print(f"[PROCESSING] 'untransfected' detected for {image_file}. Processing {alexa_image_file} only.")
                # If the image is blank, then read the alexa image and analyze it directly
                image_alexa = io.imread(image_path_alexa)

                # Convert second image to grayscale and apply hysteresis thresholding
                image_alexa_gray = color.rgb2gray(image_alexa)
                binary_image_alexa = enhance_honeycomb(image_alexa_gray)
                binary_image_alexa = (binary_image_alexa > 0).astype(np.uint8) * 255

                analyzer_alexa = JunctionsAnalyzer(image_alexa)
                binary_alexa_foreground = analyzer_alexa.preprocess_image()
                nodes, edges = analyzer_alexa.detect_nodes_edges(binary_alexa_foreground)

                selected_edges = edges if len(edges) <= 10 else random.sample(edges, edge_count)
                intersecting_edges_info = []  # Initialize the list to hold edge info
                for index, edge in enumerate(selected_edges):
                    # Calculate the properties for each selected edge
                    edge_centroid = calculate_edge_centroid(edge)
                    edge_length = calculate_edge_length(edge)
                    edge_intensity = calculate_edge_intensity(edge, image_alexa)  # Ensure you are using the correct image

                    # Append the edge info to the list, including the index as a label
                    intersecting_edges_info.append({
                        'label': index,  # Use the loop index as the edge label
                        'centroid': edge_centroid,
                        'length': edge_length,
                        'average_intensity': edge_intensity
                    })

                normalizing_value = 1  # Normalization is not needed, given untransfected is assessed overall
                
                # After processing the selected edges, prepare them for CSV output
                all_edges_info.extend(prepare_edges_info_for_csv(intersecting_edges_info, normalizing_value, image_path_alexa))

            else:
                enhanced_image = enhance_honeycomb(image_gray)
                binary_image = (enhanced_image > 0).astype(np.uint8) * 255
                inverted_image = cv2.bitwise_not(binary_image)

                # Find contours and hierarchy
                contours, hierarchy = cv2.findContours(inverted_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                # Calculate the centers of the contours
                centers = []
                for i in range(len(contours)):
                    if hierarchy[0, i, 3] < 0:  # Only for outer contours
                        M = cv2.moments(contours[i])
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            centers.append((cX, cY))

                # Ensure there are centers before clustering
                if centers:
                    # Hierarchical clustering to find clusters of centers
                    min_centroids_per_cluster = 3  # Decrease if small clusters are being ignored
                    min_area_threshold = 70  # Decrease to include clusters with smaller bounding boxes
                    aspect_ratio_threshold = 20  # Increase if elongated clusters are being discarded

                    proximity_threshold = 60  # Increase if points that should be clustered together are not
                    clusters = fclusterdata(np.array(centers), t=proximity_threshold, criterion='distance')

                    # Count the number of centers in each cluster
                    cluster_counts = np.bincount(clusters)

                    # New filtering logic replaces the old min_cluster_size check
                    filtered_centers = []
                    for idx, center in enumerate(centers):
                        cluster_index = clusters[idx]
                        cluster_size = cluster_counts[cluster_index]

                        # Retrieve all centroids in the same cluster
                        cluster_centroids = [centers[i] for i, c_idx in enumerate(clusters) if c_idx == cluster_index]
                        
                        # Calculate the bounding box for the cluster
                        x_coords, y_coords = zip(*cluster_centroids)
                        min_x, max_x, min_y, max_y = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
                        width, height = max_x - min_x, max_y - min_y
                        
                        # Calculate area of bounding box and aspect ratio
                        area = width * height
                        aspect_ratio = width / height if height > 0 else 0
                        
                        # Apply additional filtering criteria
                        if (cluster_size >= min_centroids_per_cluster and
                            area >= min_area_threshold and
                            aspect_ratio <= aspect_ratio_threshold):
                            
                            # Add to filtered list if all criteria are met
                            filtered_centers.extend(cluster_centroids)

                    # Derive edges from filtered centers
                    edges = derive_edges_from_centers(filtered_centers)
                    if len(edges) > edge_count:
                        edges = random.sample(edges, edge_count) 

                    # Create a mask with circles drawn around each filtered central point
                    radius = 40  # Adjust radius as necessary
                    mask = np.zeros_like(image_gray, dtype=np.uint8)
                    for (cX, cY) in filtered_centers:
                        cv2.circle(mask, (cX, cY), radius, 255, -1)
                else:
                    print("No centers found. Please check contour detection.")

                # Convert second image to grayscale and apply hysteresis thresholding
                image_alexa_gray = color.rgb2gray(image_alexa)
                binary_image_alexa = enhance_honeycomb(image_alexa_gray)
                binary_image_alexa = (binary_image_alexa > 0).astype(np.uint8) * 255

                # Use the binary mask to extract the foreground regions
                foreground = cv2.bitwise_and(image_alexa, image_alexa, mask=mask)
                alexa_foreground = cv2.bitwise_and(binary_image_alexa, binary_image_alexa, mask=mask)

                # Binarize the extracted foreground
                foreground_gray = color.rgb2gray(foreground) if foreground.ndim == 3 else foreground
                _, binary_foreground = cv2.threshold(foreground_gray, 1, 255, cv2.THRESH_BINARY)

                # Perform skeletonization with pruning on alexa_foreground
                skeleton_alexa_foreground = skeletonize(alexa_foreground > 0)
                skeleton_alexa_foreground_pruned = remove_small_objects(skeleton_alexa_foreground, min_size=0)
                skeleton_alexa_foreground_pruned = util.img_as_ubyte(skeleton_alexa_foreground_pruned)

                alexa_foreground_rgb = color.gray2rgb(alexa_foreground)

                centroid_pairs = find_nearest_pairs(filtered_centers)
                mark_intersections(alexa_foreground_rgb, centroid_pairs, skeleton_alexa_foreground_pruned)

                # Get normalization values by getting rid of all YFP regions from ALEXA image
                modified_alexa = apply_inverted_mask(image_alexa, inverted_image)
                analyzer = JunctionsAnalyzer(modified_alexa)
                binary_alexa_foreground = analyzer.preprocess_image()
                _, edges = analyzer.detect_nodes_edges(binary_alexa_foreground)

                # Apply JunctionsAnalyzer to alexa_foreground
                print("Applying JunctionsAnalyzer to alexa_foreground...")
                analyzer = JunctionsAnalyzer(alexa_foreground)
                binary_alexa_foreground = analyzer.preprocess_image()
                nodes, edges = analyzer.detect_nodes_edges(binary_alexa_foreground)

                analyzer.visualize_nodes_edges_with_labels()

                # Extract contours from the alexa_foreground image
                contours, _ = cv2.findContours(alexa_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Calculate centroids
                centroids = calculate_centroids(contours)

                # Find nearest pairs of centroids
                centroid_pairs = find_nearest_pairs(centroids)

                # Now you can use the centroid_pairs for the mark_intersections function
                intersections = mark_intersections(alexa_foreground, centroid_pairs, skeleton_alexa_foreground_pruned, binary_image_alexa)

                analyzer_skeleton = JunctionsAnalyzer(skeleton_alexa_foreground_pruned)
                intersecting_nodes, intersecting_edges = analyzer_skeleton.extract_intersecting_parts(intersections)

                # Define a reasonable threshold for connecting centroids based on your specific data
                proximity_threshold = 100  # for example, adjust this value as needed

                # Apply the function to image
                connect_centroids_within_threshold(alexa_foreground_rgb, filtered_centers, proximity_threshold)

                # Use this to draw and get the yellow lines
                yellow_lines = connect_centroids_within_threshold(alexa_foreground_rgb, filtered_centers, proximity_threshold)

                overlay_image_with_lines = create_overlay_image(alexa_foreground_rgb, yellow_lines)

                # Extract only the edges that intersect with the centroid lines (yellow lines)
                intersecting_edges, intersecting_edges_info = extract_edges_intersecting_with_lines(edges, yellow_lines, foreground)

                normalizing_value = calculate_moving_average_intensity(edges, modified_alexa, edge_count)

                if len(edges) > edge_count:
                    selected_edges = random.sample(edges, edge_count)
                else:
                    selected_edges = edges

                intersecting_edges_info = []

                for i, edge in enumerate(selected_edges):
                    # Append the edge info to the list
                    edge_properties = {
                        'image_name': filename,
                        'protein_name': protein_name,
                        'condition_name': condition_name,
                        'experiment_no': experiment_no,
                        'image_no': image_no,
                        'centroid_x': calculate_edge_centroid(edge),
                        'centroid_y': calculate_edge_centroid(edge),
                        'length': calculate_edge_length(edge),
                        'average_intensity': calculate_edge_intensity(edge, image_alexa),
                        'label': i
                    }

                    intersecting_edges_info.append(edge_properties)

                prepared_edge_data = prepare_edges_info_for_csv(intersecting_edges_info, normalizing_value, image_path)
                all_edges_info.extend(prepared_edge_data)

                print(f"[FINISHED PROCESSING IMAGE]")
                print(f"###########################")
        except Exception as e:
            print(f"Error processing images: {e}")
            pass

    csv_filepath = os.path.join(csv_output_dir, user_csv_filename + f'_{edge_count}.csv')
    df = pd.DataFrame(all_edges_info)
    df.to_csv(csv_filepath, index=False)
    print(f"[CSV SAVED]")

for edge_count in range(20, 21):
    execute_main(edge_count)
