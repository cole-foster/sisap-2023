# SISAP 2023 Indexing Challenge
**Team Name:** HSP

#### Description
A hierarchical, graph-based approximate search method. The key idea is to select pivots that are properly distributed 
over the dataset, ensuring quality starting points over the bottom-layer graph. The pivots on the upper layers are 
connected by an approximately monotonic graph, the approximate HSP Graph, which help improve fast graph traversal to 
starting points. Multiple search methods are tested, utilizing these well distributed pivots for diverse yet optimal 
starting points. 

This code heavily relies on the highly-optimized implementation of the Hierarchical Navigable Small World (HNSW) by
[yurymalkov](https://github.com/nmslib/hnswlib). 

~~~
    @article{malkov2018efficient,
    title={Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs},
    author={Malkov, Yu A and Yashunin, Dmitry A},
    journal={IEEE transactions on pattern analysis and machine intelligence},
    volume={42},
    number={4},
    pages={824--836},
    year={2018},
    publisher={IEEE}
    }
~~~

#### Environment Setup
- Python/3.8
- 




