#include "../../hnswlib/hnswlib.h"


int main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10000;   // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::L2Space space(dim);
    // hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, "hnsw-10k.bin");

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // need to add this for it to work
    size_t total_size_bytes = max_elements*dim*sizeof(float);
    char* dataPointer_ = (char *) malloc(total_size_bytes);
    memcpy(dataPointer_, data, total_size_bytes);
    alg_hnsw->data_pointer_ = dataPointer_;

 


    // call the pivot selection thing

    alg_hnsw->selectPivotsAndComputeHSP(2, 1.6, 100, 10);

   // // Add data to index
    // for (int i = 0; i < max_elements; i++) {
    //     alg_hnsw->addPoint(data + i * dim, i);
    // }
    // Query the elements for themselves and measure recall
    // float correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // float recall = correct / max_elements;
    // std::cout << "Recall: " << recall << "\n";
    // Serialize index
    // std::string hnsw_path = "hnsw-10k.bin";
    // alg_hnsw->saveIndex(hnsw_path);
    // delete alg_hnsw;
    // // Deserialize index and check recall
    // alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
    // correct = 0;
    // for (int i = 0; i < max_elements; i++) {
    //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
    //     hnswlib::labeltype label = result.top().second;
    //     if (label == i) correct++;
    // }
    // recall = (float)correct / max_elements;
    // std::cout << "Recall of deserialized index: " << recall << "\n";

    free(dataPointer_);
    delete[] data;
    delete alg_hnsw;
    return 0;
}
