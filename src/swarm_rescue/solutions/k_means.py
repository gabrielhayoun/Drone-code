import numpy as np

class KMeans:

    def __init__(self, dataset, SAMPLE_SIZE=100):

        sample_indexes = np.arange(len(dataset))
        np.random.shuffle(sample_indexes)

        self._dataset = np.array([dataset[sample_indexes[i]] for i in range(min(SAMPLE_SIZE, len(dataset)))],
                                 dtype=float)
        self._dataset_size = len(self._dataset)

    def process(self, k_max, k_min=1):

        centroids = []
        distortions = []

        for k in range(k_min, min(k_max, self._dataset_size)+1):

            result = self._k_means_single(k)

            centroids.append(result[0])
            distortions.append(result[1])

        return centroids[np.argmin(distortions)]

    def _k_means_single(self,k):
        # Initialization

        centroids = self._initial_centroids_generator(k)
        associated_centroids = np.zeros(self._dataset_size, dtype=int)

        previous_distortion_function = np.Infinity

        p = 0

        while previous_distortion_function - self._distortion_function(centroids, associated_centroids) > 0:

            previous_distortion_function = self._distortion_function(centroids, associated_centroids)

            # Computing closest centroids
            associated_centroids = [np.argmin(np.linalg.norm(centroids - self._dataset[i], axis=-1))
                                    for i in range(self._dataset_size)]

            # Computing new values of centroids
            for j in range(k):

                associated_data = []

                for i in range(self._dataset_size):

                    if associated_centroids[i] == j:
                        associated_data.append(self._dataset[i])

                centroids[j] = np.sum(associated_data, axis=0) / len(associated_data)

            p += 1

        return centroids, self._distortion_function(centroids, associated_centroids)

    def _initial_centroids_generator(self, k):

        if k > self._dataset_size:
            print("initialCentroidsIndexesGenerator Error : n<k")
            return -1

        initial_centroids_indexes = []

        while len(initial_centroids_indexes) < k:

            index = np.random.randint(self._dataset_size)

            if index not in initial_centroids_indexes:
                initial_centroids_indexes.append(index)

        return np.array([self._dataset[i] for i in initial_centroids_indexes])

    def _distortion_function(self, centroids, associated_centroids):
        s = 0

        for i in range(self._dataset_size):
            s += np.linalg.norm(self._dataset[i] - centroids[associated_centroids[i]]) ** 2

        return s / self._dataset_size



"""if __name__ == "__main__":
    A = np.random.random((50,2))
    print(A)
    kmeans = KMeans(A)

    print("result",kmeans.process(5))"""