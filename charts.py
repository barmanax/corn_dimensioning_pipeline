import json
import matplotlib.pyplot as plt
import numpy as np

def create_charts(data):
    names = [entry['name'] for entry in data]
    avg_widths = [entry['avg_width'] for entry in data]
    avg_heights = [entry['avg_height'] for entry in data]
    max_widths = [entry['max_width'] for entry in data]
    max_heights = [entry['max_height'] for entry in data]

    # Plot average height vs width of images
    plt.figure(figsize=(12, 6))
    plt.scatter(avg_heights, avg_widths, color='blue')
    for i, name in enumerate(names):
        plt.annotate(name, (avg_heights[i], avg_widths[i]), fontsize=8, alpha=0.7)
    plt.title('Average Heights vs Width of Images')
    plt.xlabel('Average Heights')
    plt.ylabel('Average Width')
    plt.xticks(rotation=90)
    plt.xlim(min(avg_heights) * 0.9, max(avg_heights) * 1.1)  # Zoom out on x-axis
    plt.ylim(min(avg_widths) * 0.9, max(avg_widths) * 1.1)   # Zoom out on y-axis
    plt.tight_layout()
    plt.savefig('charts/avg_height_vs_width_scatter.png')
    plt.show()

    # Plot max height vs width of images
    plt.figure(figsize=(12, 6))
    plt.scatter(max_heights, max_widths, color='green')
    for i, name in enumerate(names):
        plt.annotate(name, (max_heights[i], max_widths[i]), fontsize=8, alpha=0.7)
    plt.title('Max Height vs Width of Images')
    plt.xlabel('Max Height')
    plt.ylabel('Max Width')
    plt.xticks(rotation=90)
    plt.xlim(min(max_heights) * 0.9, max(max_heights) * 1.1)  # Zoom out on x-axis
    plt.ylim(min(max_widths) * 0.9, max(max_widths) * 1.1)   # Zoom out on y-axis
    plt.tight_layout()
    plt.savefig('charts/max_height_vs_width_scatter.png')
    plt.show()

    # Create bar charts for each category with 5 buckets
    def create_bar_chart(data, title, xlabel, ylabel, filename):
        # Calculate bucket ranges
        min_val = min(data)
        max_val = max(data)
        bucket_edges = np.linspace(min_val, max_val, 6)  # 5 buckets -> 6 edges

        # Count entries in each bucket
        counts, _ = np.histogram(data, bins=bucket_edges)

        # Create bar chart
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(counts)), counts, tick_label=[f'{bucket_edges[i]:.2f} - {bucket_edges[i+1]:.2f}' for i in range(len(counts))])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()

    # Bar charts for average width, average height, max width, and max height
    create_bar_chart(avg_widths, 'Distribution of Average Widths', 'Lengths(in)', 'Count', 'charts/avg_width_bars.png')
    create_bar_chart(avg_heights, 'Distribution of Average Heights', 'Lengths(in)', 'Count', 'charts/avg_height_bars.png')
    create_bar_chart(max_widths, 'Distribution of Max Widths', 'Lengths(in)', 'Count', 'charts/max_width_bars.png')
    create_bar_chart(max_heights, 'Distribution of Max Heights', 'Lengths(in)', 'Count', 'charts/max_height_bars.png')


if __name__ == "__main__":
    # Opening JSON file
    f = open('image_data.json')

    # returns JSON object as 
    # a dictionary
    data = json.load(f)

    misdetections = []


    for entry in data:
        if (entry['avg_height'] < 500) and (entry['name'][:-2] not in misdetections):
            misdetections.append(entry['name'][:-2])
            # print(entry['name'])


    print("Likely Misdetections: " + str((len(misdetections) / len(data)) * 100) + str("%") )
    for i in misdetections : 
        print(i)

    # Closing file
    f.close()

    create_charts(data)

    