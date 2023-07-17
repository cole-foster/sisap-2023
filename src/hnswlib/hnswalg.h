#pragma once

#include <assert.h>
#include <omp.h>
#include <stdlib.h>

#include <atomic>
#include <chrono>
#include <list>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "hnswlib.h"
#include "visited_list_pool.h"

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
   public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{0};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    VisitedListPool *visited_list_pool_{nullptr};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{0};

    char *data_level0_memory_{nullptr};  // only stores links/labels now- no data
    char **linkLists_{nullptr};          // stores upper-layer links
    std::vector<int> element_levels_;    // keeps level of each element

    // MODIFIED: POINTER TO HOLD ALL DATA. OWNED BY PYTHON.
    size_t data_size_{0};
    char *data_pointer_{nullptr};

    // distance function from space
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    mutable dist_t averageEntryPointDistance_{0};
    mutable dist_t averageClosestPivotDistance_{0};

    std::vector<std::vector<tableint>> pivot_index_vectors{};

    // deleted functionality removed...
    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions
    std::mutex deleted_elements_lock;     // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements

    HierarchicalNSW(SpaceInterface<dist_t> *s) {}

    HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false,
                    size_t max_elements = 0, bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200,
                    size_t random_seed = 100, bool allow_replace_deleted = false)
        : link_list_locks_(max_elements),
          label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
          element_levels_(max_elements),
          allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        M_ = M;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        // MODIFICATIONS
        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + sizeof(labeltype);
        // old:     size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;  // this isn't used anymore
        label_offset_ = size_links_level0_;
        // old:     label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *)malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr) throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = new VisitedListPool(1, max_elements);

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    ~HierarchicalNSW() {
        free(data_level0_memory_);
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0) free(linkLists_[i]);
        }
        free(linkLists_);
        delete visited_list_pool_;
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) { ef_ = ef; }

    inline std::mutex &getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_),
               sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *)(data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }

    // this gives data representation of the point
    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_pointer_ + internal_id * data_size_);
        // return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
    }

    size_t getMaxElements() { return max_elements_; }

    size_t getCurrentElementCount() { return cur_element_count; }

    size_t getDeletedCount() { return num_deleted_; }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidateSet;

        dist_t lowerBound;
        // if (!isMarkedDeleted(ep_id)) {
        dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
        top_candidates.emplace(dist, ep_id);
        lowerBound = dist;
        candidateSet.emplace(-dist, ep_id);
        // } else {
        //     lowerBound = std::numeric_limits<dist_t>::max();
        //     candidateSet.emplace(-lowerBound, ep_id);
        // }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock<std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int *)get_linklist0(curNodeNum);
            } else {
                data = (int *)get_linklist(curNodeNum, layer);
                //                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint *)data);
            tableint *datal = (tableint *)(data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    // if (!isMarkedDeleted(candidate_id)) top_candidates.emplace(dist1, candidate_id);
                    top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_) top_candidates.pop();

                    if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    /**
     * @brief Original Bottom Layer Search- not used
     *
     * @tparam has_deletions
     * @tparam collect_metrics
     * @param ep_id
     * @param data_point
     * @param ef
     * @param isIdAllowed
     * @return std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
     * CompareByFirst>
     */
    template <bool has_deletions, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef,
                      BaseFilterFunctor *isIdAllowed = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidate_set;

        dist_t lowerBound;
        if ((!has_deletions || !isMarkedDeleted(ep_id)) &&
            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound &&
                (top_candidates.size() == ef || (!isIdAllowed && !has_deletions))) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);
            //                bool cur_node_deleted = isMarkedDeleted(current_node_id);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                             _MM_HINT_T0);  ////////////
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                         offsetLevel0_,  ///////////
                                     _MM_HINT_T0);       ////////////////////////
#endif

                        if ((!has_deletions || !isMarkedDeleted(candidate_id)) &&
                            ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))
                            top_candidates.emplace(dist, candidate_id);

                        if (top_candidates.size() > ef) top_candidates.pop();

                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    // literally the hsp test
    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            &top_candidates,
        const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M) break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist = fstdistfunc_(getDataByInternalId(second_pair.second),
                                              getDataByInternalId(curent_pair.second), dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *)(data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *)(linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    tableint mutuallyConnectNewElement(
        const void *data_point, tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            &top_candidates,
        int level, bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *)(ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate) throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock<std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax) throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c) throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *)(ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to
            // modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                        CompareByFirst>
                        candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(fstdistfunc_(getDataByInternalId(data[j]),
                                                        getDataByInternalId(selectedNeighbors[idx]), dist_func_param_),
                                           data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        delete visited_list_pool_;
        visited_list_pool_ = new VisitedListPool(1, new_max_elements);

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char *data_level0_memory_new = (char *)realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char **linkLists_new = (char **)realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize) output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open()) throw std::runtime_error("Cannot open file");

        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count) max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize) throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *)malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_ = new VisitedListPool(1, max_elements);

        linkLists_ = (char **)malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *)malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    template <typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        // if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char *data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *)dist_func_param_);
        std::vector<data_t> data;
        data_t *data_ptr = (data_t *)data_ptrv;
        for (int i = 0; i < (int)dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    /**
     * ===========================================================
     *
     *          KEEPING DELETED FUNCTIONALITY, BUT NOT USED
     *
     * ===========================================================
     */

    /*
     * Marks an element with the given label deleted, does NOT really change the current graph.
     */
    void markDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }

    /*
     * Uses the last 16 bits of the memory for the linked list size to store the mark,
     * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
     */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    /*
     * Removes the deleted mark of the node, does NOT really change the current graph.
     *
     * Note: the method is not safe to use when replacement of deleted elements is enabled,
     *  because elements marked as deleted can be completely removed by addPoint
     */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
        std::unique_lock<std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock<std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    /*
     * Remove the deleted mark of the node.
     */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock<std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }

    /*
     * Checks the first 16 bits of the memory to see if the element is marked deleted.
     */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    unsigned short int getListCount(linklistsizeint *ptr) const { return *((unsigned short int *)ptr); }

    void setListCount(linklistsizeint *ptr, unsigned short int size) const {
        *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
    }

    /**
     * ======================================================
     *
     *          REMOVED FUNCTIONS FOR UPDATES
     *
     * ======================================================
     */

    /*
     * Adds point. Updates the point if it is already in the index.
     * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new
     * point
     */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }
        addPoint(data_point, label, -1);
    }

    /**
     * @brief Add Point To The Level
     *
     * @param data_point
     * @param label
     * @param level
     * @return tableint
     */
    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock<std::mutex> lock_table(label_lookup_lock);

            // EDIT: REMOVED CHECK FOR DELETED FUNCTIONALITY
            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0) curlevel = level;
        element_levels_[cur_c] = curlevel;

        std::unique_lock<std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy) templock.unlock();
        tableint currObj = enterpoint_node_;

        // set the memory for this element
        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        // memcpy(getDataByInternalId(cur_c), data_point, data_size_); // REMOVED: not storing memory anymore

        if (curlevel) {
            linkLists_[cur_c] = (char *)malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock<std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *)(data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_) throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst>
                    top_candidates = searchBaseLayer(currObj, data_point, level);
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    void checkIntegrity() {
        int connections_checked = 0;
        std::vector<int> inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *)(ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] > 0);
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i = 0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    /**
     * @brief ORIGINAL HNSW SEARCH FUNCTION
     *
     * @param query_data
     * @param k
     * @param isIdAllowed
     * @return std::priority_queue<std::pair<dist_t, labeltype>>
     */
    std::priority_queue<std::pair<dist_t, labeltype>> searchKnn(const void *query_data, size_t k,
                                                                BaseFilterFunctor *isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;
                // printf(" Level %d, Point %u, NN: %d\n", level, currObj, size);

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    // printf("  *  %u\n", cand);

                    if (cand < 0 || cand > max_elements_) {
                        printf(" Level %d, Point %u, cand: %u\n", level, currObj, (unsigned int)cand);
                        throw std::runtime_error("cand error");
                    }
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;
        if (num_deleted_) {
            top_candidates = searchBaseLayerST<true, true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false, true>(currObj, query_data, std::max(ef_, k), isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }


    /*
    ====================================================================================================================
    |
    |
    |
                        Final Modifications
    |
    |
    |
    ====================================================================================================================
    */

    /**
     * @brief
     *
     */
    void deletePivotsAndLinks() {
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0) {
                free(linkLists_[i]);
            }
        }
        // free(linkLists_);
    }

    /**
     * @brief
     *
     * @param level
     * @param pivotsList
     */
    void getPivotsInLevel(int const level, std::vector<tableint> &pivotsList) const {
        if (level == 0) {
            printf("Level: %d is just the bottom, not returning all points\n", level);
            return;
        }
        if (level > maxlevel_) {
            printf("Max Level is: %d\n", maxlevel_);
            return;
        }

        // get the points in the layer
        pivotsList.clear();
        for (tableint i = 0; i < (tableint)cur_element_count; i++) {
            if (element_levels_[i] >= level) {
                pivotsList.push_back(i);
            }
        }

        return;
    }

    /**
     * @brief
     *
     */
    void printHierarchy() {
        printf("Printing the number of points on each level:\n");
        for (int i = maxlevel_; i >= 1; i--) {
            std::vector<tableint> pointsInLevel{};
            getPivotsInLevel(i, pointsInLevel);
            printf("  L-%d: %u\n", i, (unsigned int)pointsInLevel.size());
        }
        printf("  L-0: %u\n", (unsigned int)cur_element_count);
    }

    /**
     * @brief
     *
     * @param level
     * @param pivot_index
     * @param neighbors
     */
    void getNeighborsInLevel(int const level, tableint const pivot_index, std::vector<tableint> &neighbors) const {
        if (level <= 0 || level > maxlevel_) {
            printf("Can Only Change Neighbors of Pivots in Levels 1...%d\n", (int)maxlevel_);
            return;
        }
        neighbors.clear();

        // get pointer to the list of neighbors.
        linklistsizeint *ll_cur = get_linklist(pivot_index, level);  // top level is 2

        // initialize this pointer with however many neighbors there actually are
        int size_neighbors = getListCount(ll_cur);

        // format the pointer of this
        tableint *data = (tableint *)(ll_cur + 1);

        // add the neighbors!
        for (int it2 = 0; it2 < size_neighbors; it2++) {
            neighbors.push_back((tableint)data[it2]);
        }

        return;
    }

    /**
     * @brief
     *
     * @param level
     * @param pivot_index
     * @param neighbors
     */
    void setNeighborsInLevel(int const level, tableint const pivot_index, std::vector<tableint> const &neighbors) {
        if (level <= 0 || level > maxlevel_) {
            printf("Can Only Change Neighbors of Pivots in Levels 1...%d\n", (int)maxlevel_);
            return;
        }

        // can only have up to max_neighbors;
        int const size_neighbors = (int)std::min(neighbors.size(), maxM_);

        // get pointer to the list of neighbors.
        linklistsizeint *ll_cur = get_linklist(pivot_index, level);  // top level is 2

        // initialize this pointer with however many neighbors there actually are
        setListCount(ll_cur, (unsigned short)size_neighbors);

        // format the pointer of this
        tableint *data = (tableint *)(ll_cur + 1);

        // add the neighbors!
        for (int it2 = 0; it2 < size_neighbors; it2++) {
            data[it2] = (tableint)neighbors[it2];
        }

        return;
    }

    /**
     * @brief
     *
     * @param queryIndex
     * @param pivots_list
     * @param max_hsp_neighborhood_size
     * @param neighbors
     */
    void HSP_Test(tableint const queryIndex, std::vector<tableint> const &pivots_list,
                  int const max_hsp_neighborhood_size, std::vector<tableint> &neighbors) {
        neighbors.clear();
        char *q_data = getDataByInternalId(queryIndex);

        // only perform on k closest elements
        std::vector<std::pair<dist_t, tableint>> L{};

        // find next nearest neighbor and create list of distances
        tableint index1;
        dist_t distance_Q1 = HUGE_VAL;
        for (int it1 = 0; it1 < (int)pivots_list.size(); it1++) {
            tableint const index = pivots_list[it1];
            if (index == queryIndex) continue;
            dist_t const d = fstdistfunc_(q_data, getDataByInternalId(index), dist_func_param_);
            if (d < distance_Q1) {
                distance_Q1 = d;
                index1 = index;
            }
            L.emplace_back(d, index);
        }

        // only want to perform hsp algo on top k neighbors
        if ((max_hsp_neighborhood_size > 0) && (max_hsp_neighborhood_size < (int)pivots_list.size())) {
            typename std::vector<std::pair<dist_t, tableint>>::iterator position_to_sort =
                L.begin() + max_hsp_neighborhood_size;
            std::nth_element(L.begin(), position_to_sort, L.end());
            while (L.size() > (size_t)max_hsp_neighborhood_size) L.pop_back();
        }

        // now, eliminate points and find next hsp neighbors
        while (L.size() > 0) {
            // adding the new neighbor
            neighbors.push_back(index1);

            // prepare for elimination, and for finding next neighbor
            char *index1_data = getDataByInternalId(index1);
            std::vector<std::pair<dist_t, tableint>> L_copy = L;
            L.clear();
            tableint index1_next;
            dist_t distance_Q1_next = HUGE_VAL;

            // compute distances in parallel
            for (int it1 = 0; it1 < (int)L_copy.size(); it1++) {
                tableint const index2 = L_copy[it1].second;
                if (index2 == index1 || index2 == queryIndex) continue;
                dist_t const distance_Q2 = L_copy[it1].first;
                dist_t const distance_12 = fstdistfunc_(index1_data, getDataByInternalId(index2), dist_func_param_);

                // check inequalities
                if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    L.emplace_back(distance_Q2, index2);
                    if (distance_Q2 < distance_Q1_next) {
                        distance_Q1_next = distance_Q2;
                        index1_next = index2;
                    }
                }
            }

            // setup the next hsp neighbor
            index1 = index1_next;
            distance_Q1 = distance_Q1_next;
        }

        return;
    }

    /**
     * @brief
     *
     * @param queryIndex
     * @param pivots_list
     * @param max_hsp_neighborhood_size
     * @param neighbors
     */
    void kNN_Test(tableint const queryIndex, std::vector<tableint> const &pivots_list, int const k,
                  std::vector<tableint> &neighbors) {
        neighbors.clear();
        char *q_data = getDataByInternalId(queryIndex);

        // only perform on k closest elements
        std::vector<std::pair<dist_t, tableint>> L{};

        // find next nearest neighbor and create list of distances
        tableint index1;
        dist_t distance_Q1 = HUGE_VAL;
        for (int it1 = 0; it1 < (int)pivots_list.size(); it1++) {
            tableint const index = pivots_list[it1];
            if (index == queryIndex) continue;
            dist_t const d = fstdistfunc_(q_data, getDataByInternalId(index), dist_func_param_);
            if (d < distance_Q1) {
                distance_Q1 = d;
                index1 = index;
            }
            L.emplace_back(d, index);
        }

        // only want to perform hsp algo on top k neighbors
        if ((k > 0) && (k < (int)pivots_list.size())) {
            typename std::vector<std::pair<dist_t, tableint>>::iterator position_to_sort = L.begin() + k;
            std::nth_element(L.begin(), position_to_sort, L.end());
            while (L.size() > (size_t)k) L.pop_back();
        }

        for (int it1 = 0; it1 < L.size(); it1++) {
            neighbors.push_back(L[it1].second);
        }

        return;
    }

    /**
     * @brief Create the Monotonic Hierarchy
     *
     * Takes the HNSW pivots as input, aims to create a monotonic on each layer of the pivots
     */
    void createMonotonicHierarchy() {
        //  - parameters
        int const numThreads = (int)omp_get_max_threads();
        int const number_of_pivots_for_assisted_build = 10000;
        int const max_hsp_neighborhood_size = 10000;
        int const max_neighbors = 80;
        std::chrono::high_resolution_clock::time_point tStart, tEnd;

        //------------------------------------------------------------------------
        //>         DELETE ALL EXISING NEIGHBORS, RESET
        //------------------------------------------------------------------------

        // - delete the links for all levels of the hierarchy
        deletePivotsAndLinks();

        // - set the new max number of links
        maxM_ = (size_t)max_neighbors;
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);  // original, in constructor

        // - set the new memory for pivots
        for (tableint pivot_index = 0; pivot_index < (tableint)cur_element_count; pivot_index++) {
            if (element_levels_[pivot_index] == 0) continue;
            linkLists_[pivot_index] = (char *)malloc(size_links_per_element_ * element_levels_[pivot_index] + 1);
        }

        //------------------------------------------------------------------------
        //>         INITIALIZE THE GRAPHS ON ALL EVERY LAYER
        //------------------------------------------------------------------------
        for (int level = maxlevel_; level > 0; level--) {
            printf("Finding the HSP Neighbors on Level: %d\n", level);
            tStart = std::chrono::high_resolution_clock::now();

            //  - get the list of pivots in the level
            std::vector<tableint> pivot_list{};
            getPivotsInLevel(level, pivot_list);
            int const num_pivots = (int)pivot_list.size();
            printf("    * Number of Pivots: %d\n", num_pivots);

            //  - set number of threads to use
            int numThreadsToUse = numThreads;
            if (numThreads * 2 >= num_pivots) numThreadsToUse = 1;

            //  - initialize the hsp graph
            std::vector<std::vector<tableint>> hsp_graph{};
            hsp_graph.resize(num_pivots);

            //  - choose hsp construction type
            if (level == maxlevel_ || num_pivots <= number_of_pivots_for_assisted_build) {
                printf("    * HSP Neighbors by Brute Force\n");

                //  - compute hsp neighbors in parallel
                #pragma omp parallel for schedule(dynamic) num_threads(numThreadsToUse)
                for (int it1 = 0; it1 < num_pivots; it1++) {
                    tableint const pivot_index = pivot_list.at(it1);
                    HSP_Test(pivot_index, pivot_list, max_hsp_neighborhood_size, hsp_graph.at(it1));
                }
                printf("    * Computed all HSP Neighbors\n");

            } else {
                printf("    * HSP Neighbors by Pivot Domains\n");

                //  - get the coarse level pivots
                std::vector<tableint> coarse_pivot_list{};
                getPivotsInLevel(level + 1, coarse_pivot_list);
                int const num_coarse_pivots = (int)coarse_pivot_list.size();
                printf("    * Number of Coarse Pivots: %d\n", num_coarse_pivots);

                printf("    * Assigning Pivots to Domain of Closest Coarse Pivots...\n");
                std::chrono::high_resolution_clock::time_point tStart_A, tEnd_A;
                tStart_A = std::chrono::high_resolution_clock::now();

                //  - organize all pivots into coarse pivot domains
                std::vector<tableint> domain_assignments(num_pivots, cur_element_count+10);

                //  - assign each fine pivot to its closest coarse pivot domain
                #pragma omp parallel for schedule(static) num_threads(numThreadsToUse)
                for (int itx = 0; itx < num_pivots; itx++) {
                    tableint const pivot_index = pivot_list.at(itx);
                    char *pivot_data = getDataByInternalId(pivot_index);

                    //  - initialize the closest parent
                    tableint closestParentIndex = cur_element_count + 10;  // over num elements for verification
                    dist_t closestParentDistance = 1000000;

                    //  - test each coarse pivot
                    for (int itp = 0; itp < num_coarse_pivots; itp++) {
                        tableint const coarse_pivot_index = coarse_pivot_list.at(itp);
                        if (coarse_pivot_index == pivot_index) {
                            closestParentIndex = coarse_pivot_index;
                            break;
                        }
                        dist_t distance =
                            fstdistfunc_(pivot_data, getDataByInternalId(coarse_pivot_index), dist_func_param_);

                        // - one pivot definitely covers, so no need to check
                        if (distance < closestParentDistance) {
                            closestParentDistance = distance;
                            closestParentIndex = coarse_pivot_index;
                        }
                    }
                    domain_assignments.at(itx) = closestParentIndex;
                }

                //  - assign to domains
                std::vector<std::vector<tableint>> coarse_pivot_domains{};
                coarse_pivot_domains.resize(num_coarse_pivots);
                for (int itp = 0; itp < num_coarse_pivots; itp++) {
                    tableint const coarse_pivot_index = coarse_pivot_list.at(itp);

                    //  - check all points for belonging to its domain
                    for (int itx = 0; itx < num_pivots; itx++) {
                        tableint const pivot_index = pivot_list.at(itx);
                        if (domain_assignments.at(itx) == coarse_pivot_index) {
                            coarse_pivot_domains.at(itp).push_back(pivot_index);
                        }
                    }
                }
                tEnd_A = std::chrono::high_resolution_clock::now();
                double time_A = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd_A - tStart_A).count();
                printf("    * Done assigning pivots in %.3f seconds\n", time_A);

                //  - collect some stats
                int minDomainSize = cur_element_count + 10;
                int maxDomainSize = 0;
                double totalDomainSize = 0;
                for (int itp = 0; itp < num_coarse_pivots; itp++) {
                    std::vector<tableint> const &coarse_domain = coarse_pivot_domains.at(itp);
                    int const domain_size = (int)coarse_domain.size();
                    if (domain_size < minDomainSize) minDomainSize = domain_size;
                    if (domain_size > maxDomainSize) maxDomainSize = domain_size;
                    totalDomainSize += (double)domain_size;
                }
                printf("    - Domain Stats:\n");
                printf("        * Total Number Of Domains: %d\n", num_coarse_pivots);
                printf("        * Total Number Of Pivots: %d\n", num_pivots);
                printf("        * Min Domain Size: %d\n", minDomainSize);
                printf("        * Max Domain Size: %d\n", maxDomainSize);
                printf("        * Total Domains Size: %.0f\n", totalDomainSize);

                //  - create a map for fine pivots
                std::unordered_map<tableint, int> pivots_map{};
                for (int itp = 0; itp < num_pivots; itp++) {
                    tableint const pivot_index = pivot_list.at(itp);
                    pivots_map.emplace(pivot_index, itp);
                }

                //  - create a map for coarse pivots
                std::unordered_map<tableint, int> coarse_pivot_map{};
                for (int itp = 0; itp < num_coarse_pivots; itp++) {
                    tableint const coarse_pivot_index = coarse_pivot_list.at(itp);
                    coarse_pivot_map.emplace(coarse_pivot_index, itp);
                }

                printf("    * Computing Approximate HSP Using Pivot Domains..\n");
                std::chrono::high_resolution_clock::time_point tStart_H, tEnd_H;
                tStart_H = std::chrono::high_resolution_clock::now();

                //  - create the hsp graph
                #pragma omp parallel for schedule(dynamic) num_threads(numThreadsToUse)
                for (int itp = 0; itp < num_coarse_pivots; itp++) {
                    tableint const coarse_pivot_index = coarse_pivot_list.at(itp);
                    std::vector<tableint> const &coarse_pivot_domain = coarse_pivot_domains.at(itp);
                    std::vector<tableint> coarse_pivot_neighbors{};

                    // - print for knowing timing
                    if (itp % 10 == 0) printf("      - %d/%d\n",itp,num_coarse_pivots);

                    //  - get hsp neighbors of coarse pivot
                    std::vector<tableint> coarse_pivot_hsp_neighbors{};
                    getNeighborsInLevel(level + 1, coarse_pivot_index, coarse_pivot_hsp_neighbors);

                    //  - collect hsp neighbors of hsp neighbors
                    std::unordered_set<tableint> coarse_pivot_neighbors_of_neighbors(coarse_pivot_hsp_neighbors.begin(),
                                                                                     coarse_pivot_hsp_neighbors.end());
                    for (int itn = 0; itn < (int)coarse_pivot_hsp_neighbors.size(); itn++) {
                        tableint const neighbor_index = coarse_pivot_hsp_neighbors.at(itn);
                        std::vector<tableint> neighbor_hsp_neighbors{};
                        getNeighborsInLevel(level + 1, neighbor_index, neighbor_hsp_neighbors);
                        coarse_pivot_neighbors_of_neighbors.insert(neighbor_hsp_neighbors.begin(),
                                                                   neighbor_hsp_neighbors.end());
                    }
                    coarse_pivot_neighbors.insert(coarse_pivot_neighbors.end(),
                                                  coarse_pivot_neighbors_of_neighbors.begin(),
                                                  coarse_pivot_neighbors_of_neighbors.end());

                    //  - collect hsp domain as domains of all neighbors
                    std::vector<tableint> hsp_domain = coarse_pivot_domain;
                    for (int itn = 0; itn < (int)coarse_pivot_neighbors.size(); itn++) {
                        tableint const neighbor_index = coarse_pivot_neighbors.at(itn);
                        int const coarse_pivot_iterator = coarse_pivot_map.at(neighbor_index);
                        std::vector<tableint> const &neighbor_partition =
                            coarse_pivot_domains.at(coarse_pivot_iterator);
                        hsp_domain.insert(hsp_domain.end(), neighbor_partition.begin(), neighbor_partition.end());
                    }

                    //  - find hsp neighbors for all pivots in domain in parallel from same spotlight
                    for (int itx = 0; itx < (int)coarse_pivot_domain.size(); itx++) {
                        tableint const pivot_index = coarse_pivot_domain.at(itx);
                        int const pivot_iterator = pivots_map.at(pivot_index);

                        // the error: two of the same pivots, two to one mapping, parallel execution on same thing
                        HSP_Test(pivot_index, hsp_domain, max_hsp_neighborhood_size, hsp_graph[pivot_iterator]);
                    }
                }
                tEnd_H = std::chrono::high_resolution_clock::now();
                double time_H = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd_H - tStart_H).count();
                printf("    * Done computing HSP in %.3f seconds\n", time_H);
            }

            //  - statistics on hsp neighbors
            //  - collect some stats
            int min_hsp_neighbors = cur_element_count + 10;
            int max_hsp_neighbors = 0;
            double ave_hsp_neighbors = 0;
            for (int itp = 0; itp < num_pivots; itp++) {
                std::vector<tableint> const &hsp_neighbors = hsp_graph.at(itp);
                int const num_hsp_neighbors = (int)hsp_neighbors.size();
                if (num_hsp_neighbors < min_hsp_neighbors) min_hsp_neighbors = num_hsp_neighbors;
                if (num_hsp_neighbors > max_hsp_neighbors) max_hsp_neighbors = num_hsp_neighbors;
                ave_hsp_neighbors += (double)num_hsp_neighbors;
            }
            printf("    - HSP Neighbor Stats:\n");
            printf("        * Total Number Of Pivots: %d\n", num_pivots);
            printf("        * Min HSP Neighbors: %d\n", min_hsp_neighbors);
            printf("        * Max HSP Neighbors: %d\n", max_hsp_neighbors);
            printf("        * Ave HSP Neighbors: %.4f\n", ave_hsp_neighbors / (double)num_pivots);

            //  - set the neighbors
            #pragma omp parallel for schedule(static) num_threads(numThreadsToUse)
            for (int it1 = 0; it1 < num_pivots; it1++) {
                tableint const pivot_index = pivot_list[it1];
                std::vector<tableint> const &pivot_neighbors = hsp_graph[it1];
                setNeighborsInLevel(level, pivot_index, pivot_neighbors);
            }
            printf("    * Set All Neighbors\n");

            tEnd = std::chrono::high_resolution_clock::now();
            double time_layer = std::chrono::duration_cast<std::chrono::duration<double>>(tEnd - tStart).count();
            printf("    * Done with layer in %.3f seconds\n", time_layer);
        }

        return;
    }


    /**
     * ============================================================
     *
     *      -------------------------------------------------
     *      |                                               |
     *      |               SEARCH ALGORITHMS               |
     *      |                                               |
     *      -------------------------------------------------
     *
     * ============================================================
     */

    /**
     *
     * @brief Revised HNSW Search Algorithm: contained, no filter
     *
     * @param query_data
     * @param k
     * @return std::priority_queue<std::pair<dist_t, labeltype>>
     */
    std::priority_queue<std::pair<dist_t, labeltype>> search_hnsw(const void *query_data, size_t k) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;
        bool const collect_metrics = false;

        //> Default Entry Point
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        //> Top-Down Greedy Search for Entry-Point
        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];

                    if (cand < 0 || cand > max_elements_) {
                        throw std::runtime_error("cand error");
                    }
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        //> SEARCH THE BOTTOM LAYER
        size_t ef = std::max(ef_, k);

        // - initialize visited list for tabu search
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // - initialize lists
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;  // containing the closest visited nodes (to be size ef)
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidate_set;  // containing the list of nodes to visit

        // - initialize with bottom layer graph entry point
        dist_t dist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        dist_t lowerBound = dist;
        top_candidates.emplace(dist, currObj);
        candidate_set.emplace(-dist, currObj);
        visited_array[currObj] = visited_array_tag;

        //- STATISTICS--- NOT ATOMIC, BE CAREFUL
        // if (collect_metrics) {
        //     averageEntryPointDistance_ += lowerBound;
        //}

        // - depth-first iteratation through the list of candidate points on bottom layer
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            // termination condition: no candidate points in top_candidates, top_candidates is full
            if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef)) {
                break;
            }
            candidate_set.pop();

            // gather neighbors of current node
            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

// #ifdef USE_SSE
//             _mm_prefetch((char *)(visited_array + *(data + 1)), _MM_HINT_T0);
//             _mm_prefetch((char *)(visited_array + *(data + 1) + 64), _MM_HINT_T0);
//             _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
//             _mm_prefetch((char *)(data + 2), _MM_HINT_T0);
// #endif

            // iterate through each neighbor
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);


                // pre-fetch the information into cache...
// #ifdef USE_SSE
//                 _mm_prefetch((char *)(visited_array + *(data + j + 1)), _MM_HINT_T0);
//                 _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
//                              _MM_HINT_T0);  ////////////
// #endif

                // check if the point has been visited already! (tabu search)
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    // compute distance to the object
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);

// #ifdef USE_SSE
//                         _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
//                                          offsetLevel0_,  ///////////
//                                      _MM_HINT_T0);       ////////////////////////
// #endif

                        top_candidates.emplace(dist, candidate_id);
                        if (top_candidates.size() > ef) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        // release the visisted list
        visited_list_pool_->releaseVisitedList(vl);

        // only keep the k closest points
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }



    /**
     *
     * @brief APPROACH 2: BEAM SEARCH ON SECOND-TO-LAST LAYER FOR CLOSER ENTRY POINT
     *
     * @param query_data
     * @param k
     * @return std::priority_queue<std::pair<dist_t, labeltype>>
     */
    std::priority_queue<std::pair<dist_t, labeltype>> knn_approach2(const void *query_data, size_t k, int const m) const {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;
        bool const collect_metrics = false;

        //> Default Entry Point
        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        /*
        =============================================
        
                Greedy Search Until Level 2

        =============================================
        */

        //> Top-Down Greedy Search for Entry-Point
        for (int level = maxlevel_; level > 1; level--) {
            
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *)get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations += size;

                tableint *datal = (tableint *)(data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];

                    if (cand < 0 || cand > max_elements_) {
                        throw std::runtime_error("cand error");
                    }
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        /*
        =============================================
        
                Beam Search on Level 1

        =============================================
        */
        size_t ef = (size_t) m;

        // - initialize visited list for tabu search
        VisitedList *vl1 = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array1 = vl1->mass;
        vl_type visited_array_tag1 = vl1->curV;

        // - initialize lists
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            top_candidates;  // containing the closest visited nodes (to be size ef)
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
            candidate_set;  // containing the list of nodes to visit

        // - initialize with bottom layer graph entry point
        dist_t lowerBound = curdist;
        top_candidates.emplace(curdist, currObj);
        candidate_set.emplace(-curdist, currObj);
        visited_array1[currObj] = visited_array_tag1;

        //  - small beam search on second-to-bottom layer
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            // termination condition: no candidate points in top_candidates, top_candidates is full
            if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef)) {
                break;
            }
            candidate_set.pop();

            // gather neighbors of current node
            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist(current_node_id,1);
            size_t size = getListCount((linklistsizeint *)data);

            // iterate through each neighbor
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);

                // check if the point has been visited already! (tabu search)
                if (!(visited_array1[candidate_id] == visited_array_tag1)) {
                    visited_array1[candidate_id] = visited_array_tag1;

                    // compute distance to the object
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        top_candidates.emplace(dist, candidate_id);
                        if (top_candidates.size() > ef) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        // release the visisted list
        visited_list_pool_->releaseVisitedList(vl1);

        /*
        =============================================
        
                Beam Search on Level 0

        =============================================
        */
        ef = std::max(ef_, k);

        // - initialize visited list for tabu search
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        // - initialize candidate list with these m points
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates_copy = top_candidates;
        candidate_set = std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>{};
        while (top_candidates_copy.size() > 0) {
            std::pair<dist_t, tableint> current_node_pair = top_candidates_copy.top();
            candidate_set.emplace(-current_node_pair.first,current_node_pair.second);
            top_candidates_copy.pop();
            visited_array[current_node_pair.second] = visited_array_tag;

            if (current_node_pair.first < lowerBound) lowerBound = current_node_pair.first;
        }

        // - STATISTICS--- NOT ATOMIC, BE CAREFUL-- only use with one thread
        // if (collect_metrics) {
        //     averageEntryPointDistance_ += lowerBound;
        // }

        // - depth-first iteratation through the list of candidate points on bottom layer
        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();

            // termination condition: no candidate points in top_candidates, top_candidates is full
            if ((-current_node_pair.first) > lowerBound && (top_candidates.size() == ef)) {
                break;
            }
            candidate_set.pop();

            // gather neighbors of current node
            tableint current_node_id = current_node_pair.second;
            int *data = (int *)get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint *)data);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

            // iterate through each neighbor
            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);

                // check if the point has been visited already! (tabu search)
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    // compute distance to the object
                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(query_data, currObj1, dist_func_param_);

                    if (top_candidates.size() < ef || lowerBound > dist) {
                        candidate_set.emplace(-dist, candidate_id);
                        top_candidates.emplace(dist, candidate_id);
                        if (top_candidates.size() > ef) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        // release the visisted list
        visited_list_pool_->releaseVisitedList(vl);

        // only keep the k closest points
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    /**
     *
     * @brief HSP TO ENCODE DIRECTION FOR GREEDY RESTARTS
     *
     * @param query_data
     * @param k
     * @return std::priority_queue<std::pair<dist_t, labeltype>>
     */
    std::priority_queue<std::pair<dist_t, labeltype>> knn_approach3(const void *query_data, size_t k, int m) const {
        
        //  - initializations
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;
        int const start_level = std::min(maxlevel_,2);
        int const max_neighbors = maxM_;
        // m = std::max((int) k, m);

        //  - the knn we collect will be size m
        //  - this is the size of our neighborhood for hsp
        //  - all pivots on one layer are also pivots on layers below
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                    top_candidates;  // positive distances: pop furthest points. size m

        //>     TOP-DOWN
        for (int level = start_level; level >= 0; level--) {

            //  - initialize the neighborhood for top layer
            if (level == start_level) {

                //  - get all point in starting layer
                std::vector<tableint> const& pivots_start_layer = pivot_index_vectors[level];
                int const number_of_pivots = (int) pivots_start_layer.size();

                //  - get set of random indices, no duplicates
                std::unordered_set<int> rand_indices_set{};
                while (rand_indices_set.size() < m) {
                    int const rand_idx = rand() % (number_of_pivots - 1);
                    rand_indices_set.insert(rand_idx);
                }

                //  - initialize the list with the random pivots
                typename std::unordered_set<int>::iterator it1;
                for (it1 = rand_indices_set.begin(); it1 != rand_indices_set.end(); it1++) {
                    int const rand_idx = (*it1);
                    tableint const pivot_index = pivots_start_layer[rand_idx];
                    dist_t const distance = fstdistfunc_(query_data, getDataByInternalId(pivot_index), dist_func_param_);
                    top_candidates.emplace(distance, pivot_index);
                }
            }

            //  - get vector from priority queue, in a increasing order
            std::vector<std::pair<dist_t,tableint>> starting_neighborhood{};
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
                    top_candidates_copy = top_candidates;
            int count = top_candidates_copy.size();
            starting_neighborhood.resize(count);
            while (top_candidates_copy.size() > 0) {
                std::pair<dist_t, tableint> current_node_pair = top_candidates_copy.top();
                count--;
                starting_neighborhood[count] = current_node_pair;
                top_candidates_copy.pop();
            }
            //  - we can assume this is a sorted list :)

            //  - hsp test by sorted list
            std::vector<bool> hsp_chosen(starting_neighborhood.size(),false);
            std::vector<std::pair<dist_t,tableint>> hsp_neighbors{};
            for (int it1 = 0; it1 < starting_neighborhood.size(); it1++) {
                if (hsp_chosen[it1] == true) continue;
                hsp_chosen[it1] = true;

                //  - add the new hsp neighbor
                hsp_neighbors.push_back(starting_neighborhood[it1]);
                dist_t distance_Q1 = starting_neighborhood[it1].first;
                tableint index1 = starting_neighborhood[it1].second;
                char *index1_data = getDataByInternalId(index1);

                //  - find all points invalidated 
                for (int it2 = it1+1; it2 < (int)starting_neighborhood.size(); it2++) {
                    tableint const index2 = starting_neighborhood[it2].second;
                    if (hsp_chosen[it2] == true || index2 == index1) continue;
                    dist_t const distance_Q2 = starting_neighborhood[it2].first;
                    dist_t const distance_12 = fstdistfunc_(index1_data, getDataByInternalId(index2), dist_func_param_);

                    // if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    if (distance_12 < distance_Q2) {
                        hsp_chosen[it2] = true; 
                    }
                }
            }

            //  - initialize list for tabu search
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            //  - Use each hsp neighbor as a starting point
            for (int it1 = 0; it1 < hsp_neighbors.size(); it1++) {
                tableint currObj = hsp_neighbors[it1].second;
                dist_t curdist = hsp_neighbors[it1].first;
                if (visited_array[currObj] == visited_array_tag) continue;
                visited_array[currObj] = visited_array_tag;

                //  - perform greedy search: keep navigating until no closer neighbors
                bool changed = true;
                while (changed) {
                    changed = false;

                    //  - get the neighbors of the current node
                    unsigned int *data = (unsigned int *)get_linklist_at_level(currObj, level);
                    int const num_neighbors = (int) getListCount(data);
                    tableint *datal = (tableint *)(data + 1);

                    //  - iterate through each neighbor
                    int neighbors_to_visit = std::min(max_neighbors,num_neighbors);
                    for (int i = 0; i < neighbors_to_visit; i++) {
                        tableint cand = datal[i];
                        if (visited_array[cand] == visited_array_tag) continue;
                        visited_array[cand] = visited_array_tag;
                        dist_t distance = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        //  - update current node if possible
                        if (distance < curdist) {
                            curdist = distance;
                            currObj = cand;
                            changed = true;
                        }

                        //  - update top neighbors if possible
                        if (distance < top_candidates.top().first) {
                            top_candidates.emplace(distance, cand);
                            if (top_candidates.size() > 2*m) top_candidates.pop();
                        }
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        // only keep the k closest points
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    typedef std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> min_queue_tableint;


    /**
     *
     * @brief MERGE HSP FOR GOOD ENTRIES FOR BEAM SEARCH
     *
     * @param query_data
     * @param k
     * @return std::priority_queue<std::pair<dist_t, labeltype>>
     */
    std::priority_queue<std::pair<dist_t, labeltype>> knn_approach4(const void *query_data, size_t k, int b) const {
        
        //  - initializations
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;
        int const start_level = std::min(maxlevel_,2);
        int const max_neighbors = maxM_;
        int hsp_neighborhood_size = 2*b;

        //  - the knn we collect will be size m
        //  - this is the size of our neighborhood for hsp
        //  - all pivots on one layer are also pivots on layers below
        min_queue_tableint top_candidates;  // positive distances: pop furthest points. size m

        //=============================================================================================
        //
        //                      Multi-Start Greedy Search with HSP Entry Points
        //  
        //=============================================================================================

        //  - want the entry-points in the bottom layer
        std::vector<std::pair<dist_t,tableint>> entry_points{};

        //>     TOP-DOWN
        for (int level = start_level; level >= 0; level--) {

            //  - initialize the neighborhood for top layer
            if (level == start_level) {

                //  - get all point in starting layer
                std::vector<tableint> const& pivots_start_layer = pivot_index_vectors[level];
                int const number_of_pivots = (int) pivots_start_layer.size();

                //  - get set of random indices, no duplicates
                std::unordered_set<int> rand_indices_set{};
                while (rand_indices_set.size() < hsp_neighborhood_size) {
                    int const rand_idx = rand() % (number_of_pivots - 1);
                    rand_indices_set.insert(rand_idx);
                }

                //  - initialize the list with the random pivots
                typename std::unordered_set<int>::iterator it1;
                for (it1 = rand_indices_set.begin(); it1 != rand_indices_set.end(); it1++) {
                    int const rand_idx = (*it1);
                    tableint const pivot_index = pivots_start_layer[rand_idx];
                    dist_t const distance = fstdistfunc_(query_data, getDataByInternalId(pivot_index), dist_func_param_);
                    top_candidates.emplace(distance, pivot_index);
                }
            }

            //  - get vector from priority queue, in a increasing order
            std::vector<std::pair<dist_t,tableint>> starting_neighborhood{};
            min_queue_tableint top_candidates_copy = top_candidates;
            int count = top_candidates_copy.size();
            starting_neighborhood.resize(count);
            while (top_candidates_copy.size() > 0) {
                std::pair<dist_t, tableint> current_node_pair = top_candidates_copy.top();
                count--;
                starting_neighborhood[count] = current_node_pair;
                top_candidates_copy.pop();
            }
            //  - we can assume this is a sorted list :)

            //  - hsp test by sorted list
            std::vector<bool> hsp_chosen(starting_neighborhood.size(),false);
            std::vector<std::pair<dist_t,tableint>> hsp_neighbors{};
            for (int it1 = 0; it1 < starting_neighborhood.size(); it1++) {
                if (hsp_chosen[it1] == true) continue;
                hsp_chosen[it1] = true;

                //  - add the new hsp neighbor
                hsp_neighbors.push_back(starting_neighborhood[it1]);
                dist_t distance_Q1 = starting_neighborhood[it1].first;
                tableint index1 = starting_neighborhood[it1].second;
                char *index1_data = getDataByInternalId(index1);

                //  - find all points invalidated 
                for (int it2 = it1+1; it2 < (int)starting_neighborhood.size(); it2++) {
                    tableint const index2 = starting_neighborhood[it2].second;
                    if (hsp_chosen[it2] == true || index2 == index1) continue;
                    dist_t const distance_Q2 = starting_neighborhood[it2].first;
                    dist_t const distance_12 = fstdistfunc_(index1_data, getDataByInternalId(index2), dist_func_param_);

                    // if (distance_Q1 >= distance_Q2 || distance_12 >= distance_Q2) {
                    if (distance_12 < distance_Q2) {
                        hsp_chosen[it2] = true; 
                    }
                }
            }

            // break at bottom level for beam search
            if (level == 0) {
                entry_points = hsp_neighbors;
                break;
            }

            //  - initialize list for tabu search
            VisitedList *vl = visited_list_pool_->getFreeVisitedList();
            vl_type *visited_array = vl->mass;
            vl_type visited_array_tag = vl->curV;

            //  - Use each hsp neighbor as a starting point
            for (int it1 = 0; it1 < hsp_neighbors.size(); it1++) {
                tableint currObj = hsp_neighbors[it1].second;
                dist_t curdist = hsp_neighbors[it1].first;
                if (visited_array[currObj] == visited_array_tag) continue;
                visited_array[currObj] = visited_array_tag;

                //  - perform greedy search: keep navigating until no closer neighbors
                bool changed = true;
                while (changed) {
                    changed = false;

                    //  - get the neighbors of the current node
                    unsigned int *data = (unsigned int *)get_linklist_at_level(currObj, level);
                    int const num_neighbors = (int) getListCount(data);
                    tableint *datal = (tableint *)(data + 1);

                    //  - iterate through each neighbor
                    int neighbors_to_visit = std::min(max_neighbors,num_neighbors);
                    for (int i = 0; i < neighbors_to_visit; i++) {
                        tableint cand = datal[i];
                        if (visited_array[cand] == visited_array_tag) continue;
                        visited_array[cand] = visited_array_tag;
                        dist_t distance = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        //  - update current node if possible
                        if (distance < curdist) {
                            curdist = distance;
                            currObj = cand;
                            changed = true;
                        }

                        //  - update top neighbors if possible
                        if (distance < top_candidates.top().first) {
                            top_candidates.emplace(distance, cand);
                            if (top_candidates.size() > hsp_neighborhood_size) top_candidates.pop();
                        }
                    }
                }
            }
            visited_list_pool_->releaseVisitedList(vl);
        }

        //=============================================================================================
        //
        //                              Beam Search on Bottom Level
        //  
        //=============================================================================================
        
        //> Initialized Tabu Search
        //  - elements that have been encountered
        VisitedList *vl1 = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl1->mass;
        vl_type visited_array_tag = vl1->curV;
        //  - elements that have had their neighbors explores
        VisitedList *vl2 = visited_list_pool_->getFreeVisitedList();
        vl_type *explored_array = vl2->mass;
        vl_type explored_array_tag = vl2->curV;

        // res: the knn priority queue
        min_queue_tableint res = top_candidates;    // positive distances: pop furthest first
        while (res.size() > k) res.pop();

        // beam: initialize with the entry points from hsp
        int const max_beam_size = 8; // FIXED (int) b;
        min_queue_tableint beam;                    // negative distances: pop closest first
        for (int it1 = 0; it1 < (int) entry_points.size(); it1++) {
            beam.emplace(-entry_points[it1].first, entry_points[it1].second);
        }

        // iterate through the beam
        while (beam.size() > 0) {

            // first node
            std::pair<dist_t, tableint> current_node_pair = beam.top();
            tableint current_node_id = current_node_pair.second;
            beam.pop();

            // skip if already explored
            if (explored_array[current_node_id] == explored_array_tag) continue;
            explored_array[current_node_id] = explored_array_tag;
            visited_array[current_node_id] = visited_array_tag;

            // gather neighbors of current node
            int *data = (int *)get_linklist0(current_node_id);
            size_t num_neighbors = getListCount((linklistsizeint *)data);

            // iterate through each neighbor
            for (size_t j = 1; j <= num_neighbors; j++) {
                int neighbor_id = *(data + j);

                //  - skip if already visisted
                if (visited_array[neighbor_id] == visited_array_tag) continue;
                visited_array[neighbor_id] = visited_array_tag;
                
                // compute distance to the object
                char *neighborObj = (getDataByInternalId(neighbor_id));
                dist_t distance = fstdistfunc_(query_data, neighborObj, dist_func_param_);

                // if this can be added
                if (res.size() < k || distance < res.top().first) {
                    res.emplace(distance, neighbor_id);
                    if (res.size() > k) res.pop();
                    beam.emplace(-distance, neighbor_id);
                }
            }

            // resize the beam
            if (beam.size() > max_beam_size) {
                min_queue_tableint beam_temp;
                //  - dump all of beam into max queue
                while (beam.size() > 0) {
                    beam_temp.emplace(-beam.top().first, beam.top().second);
                    beam.pop();
                }
                //  - cut length of max queue
                while (beam_temp.size() > max_beam_size) beam_temp.pop();
                //  - return to the beam
                while (beam_temp.size() > 0) {
                    beam.emplace(-beam_temp.top().first, beam_temp.top().second);
                    beam_temp.pop();
                }
            }
        }

        // release the visisted list
        visited_list_pool_->releaseVisitedList(vl1);
        visited_list_pool_->releaseVisitedList(vl2);

        // only keep the k closest points
        while (res.size() > k) {
            res.pop();
        }
        while (res.size() > 0) {
            std::pair<dist_t, tableint> rez = res.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            res.pop();
        }
        return result;
    }
};
}  // namespace hnswlib