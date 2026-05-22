# DDoS Dataset Documentation

## Overview

This project uses the **Kaggle DDoS Dataset**, a comprehensive collection of network traffic data for DDoS attack classification and analysis. The dataset contains both normal network traffic and various types of DDoS attacks, making it suitable for training machine learning models.

## Dataset Source

- **Name**: Devendra DDoS Datasets
- **Platform**: Kaggle
- **Link**: [https://www.kaggle.com/datasets/devendra416/ddos-datasets](https://www.kaggle.com/datasets/devendra416/ddos-datasets)
- **License**: Check Kaggle dataset page for license information

## Dataset Structure

The dataset contains network traffic flows with the following characteristics:

### Size and Format

- **Format**: CSV (comma-separated values)
- **Number of Rows**: ~600,000+ network flow records
- **Number of Columns**: 83 network features
- **File Size**: Multiple gigabytes (download and extract required)

### Features (Columns)

The dataset includes various network traffic features:

#### Flow Identification
- **Flow ID** - Unique identifier for each network flow
- **Src IP** - Source IP address
- **Dst IP** - Destination IP address
- **Src Port** - Source port number
- **Dst Port** - Destination port number
- **Protocol** - Network protocol (TCP, UDP, etc.)

#### Flow Duration and Packet Information
- **Flow Duration** - Total duration of the flow (microseconds)
- **Total Fwd Packets** - Number of forward packets
- **Total Backward Packets** - Number of backward packets
- **Total Length of Fwd Packets** - Sum of forward packet lengths
- **Total Length of Bwd Packets** - Sum of backward packet lengths
- **Fwd Packet Length Max** - Maximum forward packet length
- **Fwd Packet Length Min** - Minimum forward packet length
- **Fwd Packet Length Mean** - Mean forward packet length
- **Bwd Packet Length Max** - Maximum backward packet length
- **Bwd Packet Length Min** - Minimum backward packet length
- **Bwd Packet Length Mean** - Mean backward packet length

#### Timing Information
- **Flow Bytes/s** - Bytes per second in the flow
- **Flow Packets/s** - Packets per second in the flow
- **Fwd IAT Total** - Total inter-arrival time in forward direction
- **Fwd IAT Mean** - Mean inter-arrival time in forward direction
- **Fwd IAT Std** - Standard deviation of inter-arrival time in forward direction
- **Fwd IAT Max** - Maximum inter-arrival time in forward direction
- **Fwd IAT Min** - Minimum inter-arrival time in forward direction
- **Bwd IAT Total** - Total inter-arrival time in backward direction
- **Bwd IAT Mean** - Mean inter-arrival time in backward direction
- **Bwd IAT Std** - Standard deviation of inter-arrival time in backward direction
- **Bwd IAT Max** - Maximum inter-arrival time in backward direction
- **Bwd IAT Min** - Minimum inter-arrival time in backward direction

#### TCP/IP Flags
- **Fwd PSH Flags** - Number of PSH flags in forward direction
- **Bwd PSH Flags** - Number of PSH flags in backward direction
- **Fwd URG Flags** - Number of URG flags in forward direction
- **Bwd URG Flags** - Number of URG flags in backward direction
- **Fwd Header Length** - Length of forward headers
- **Bwd Header Length** - Length of backward headers
- **Fwd Packets/s** - Forward packets per second
- **Bwd Packets/s** - Backward packets per second

#### Statistical Features
- **Min Packet Length** - Minimum packet length in flow
- **Max Packet Length** - Maximum packet length in flow
- **Packet Length Mean** - Mean packet length
- **Packet Length Std** - Standard deviation of packet length
- **Packet Length Variance** - Variance of packet length

#### Advanced Features
- **FIN Flag Count** - Number of FIN flags
- **SYN Flag Count** - Number of SYN flags
- **RST Flag Count** - Number of RST flags
- **ACK Flag Count** - Number of ACK flags
- **CWE Flag Count** - Number of CWE flags
- **ECE Flag Count** - Number of ECE flags
- **Down/Up Ratio** - Ratio of download to upload bytes
- **Average Packet Size** - Average size of packets in flow
- **Avg Fwd Segment Size** - Average forward segment size
- **Avg Bwd Segment Size** - Average backward segment size
- **Fwd Header Length** - Length of forward headers
- **Fwd Avg Bytes/Bulk** - Forward average bytes per bulk
- **Fwd Avg Packets/Bulk** - Forward average packets per bulk
- **Fwd Avg Bulk Rate** - Forward bulk rate
- **Bwd Avg Bytes/Bulk** - Backward average bytes per bulk
- **Bwd Avg Packets/Bulk** - Backward average packets per bulk
- **Bwd Avg Bulk Rate** - Backward bulk rate
- **Subflow Fwd Packets** - Forward packets in subflow
- **Subflow Fwd Bytes** - Forward bytes in subflow
- **Subflow Bwd Packets** - Backward packets in subflow
- **Subflow Bwd Bytes** - Backward bytes in subflow
- **Init Fwd Win Bytes** - Initial forward window bytes
- **Init Bwd Win Bytes** - Initial backward window bytes
- **Fwd Act Data Pkts** - Forward active data packets
- **Fwd Seg Size Min** - Forward segment size minimum
- **Active Mean** - Mean active time
- **Active Std** - Standard deviation of active time
- **Active Max** - Maximum active time
- **Active Min** - Minimum active time
- **Idle Mean** - Mean idle time
- **Idle Std** - Standard deviation of idle time
- **Idle Max** - Maximum idle time
- **Idle Min** - Minimum idle time

#### Target Variable
- **Label** - Classification label
  - `BENIGN` - Normal/legitimate traffic
  - `DDoS` - Denial of Service attack (or specific DDoS type)

### Types of Attacks

The dataset may include various DDoS attack types:
- **SYN Flood** - Exploits TCP handshake
- **UDP Flood** - Overwhelms with UDP packets
- **ICMP Flood** - Floods with ICMP packets
- **HTTP Flood** - Attacks application layer with HTTP requests
- **Slowloris** - Slow, sustained attack connections

## Data Characteristics

### Class Distribution
- **Benign Traffic** - Majority class (legitimate network flows)
- **DDoS Attacks** - Minority class (malicious flows)
- **Imbalanced Dataset** - Consider this when choosing evaluation metrics

### Feature Types
- **Numerical Features** - All 83 features are primarily numerical values
- **Categorical Features** - IP addresses and Protocol (converted to numerical in preprocessing)

### Missing Values
- Generally contains few or no missing values per original dataset design
- Our preprocessing handles potential missing values with imputation strategies

## Download Instructions

1. Visit [Kaggle DDoS Datasets](https://www.kaggle.com/datasets/devendra416/ddos-datasets)
2. Sign in with your Kaggle account (create one if needed)
3. Download the dataset CSV files
4. Extract and place files in the `data/` directory of this repository
5. Update notebook paths as needed to reference your data location

## Data Preprocessing

In this project, the dataset undergoes the following preprocessing steps:

1. **Feature Removal** - Drops non-predictive features (index, Flow ID, Timestamp)
2. **Composite Feature Splitting** - Splits IP addresses into octets for better numerical representation
3. **Missing Value Imputation** - Handles any missing values using mean/most frequent strategies
4. **Categorical Encoding** - Encodes categorical variables to numerical format
5. **Feature Standardization** - Normalizes numerical features (mean=0, std=1)

See [PREPROCESSING.md](PREPROCESSING.md) for detailed pipeline documentation.

## Related Notebooks

- `data analysis.ipynb` - Comprehensive data exploration and visualization
- `pipeline usage.ipynb` - Demonstrates preprocessing pipeline usage
- `notebooks/full_dataset_analyzer.ipynb` - Statistical analysis of the full dataset

## References and Citation

If you use this dataset in your research or publication, please cite the original authors:
- Original dataset curator: Devendra (Kaggle)
- See Kaggle dataset page for full citation and attribution details

## Additional Resources

- [CIC DDoS Attack Dataset](https://www.unb.ca/research/iscx/datasets/) - Alternative DDoS dataset
- [NSL-KDD Dataset](https://www.unb.ca/research/iscx/dataset/nsl-kdd.html) - Intrusion detection benchmark
- [UNSW-NB15 Dataset](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity-datasets/) - Modern cyber security dataset

