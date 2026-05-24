import client_label_distribution


client_label_distribution.generate_label_distribution_csv(
    output_path="client_label_distributions.csv",
    total_clients=100, partition_size=500,
    concentration=0.5, random_seed=8888,
    data_path="./data",
    partition_dist_low=0, partition_dist_high=30,
)