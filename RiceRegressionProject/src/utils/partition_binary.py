class PartitionBinary:
    def partition_binary(X, y, class_value):
        group = X[y == class_value]
        return group