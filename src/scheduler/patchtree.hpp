/**
 * @file patchtree.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief Patch tree for the mpi side of the code
 * @version 0.1
 * @date 2022-02-21
 * 
 * @copyright Copyright (c) 2022
 * 
 */


#pragma once

#include "../aliases.hpp"
#include "patch.hpp"
#include <array>
#include <cstdio>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include "../utils/geometry_utils.hpp"






struct PTNode{
    u64 x_min,y_min,z_min;
    u64 x_max,y_max,z_max;
    u64 level;

    u64 parent_id;
    u64 childs_id[8] {u64_max};

    u64 linked_patchid = u64_max;

    bool is_leaf = true;
    bool child_are_all_leafs = false;

    //patch fields
    u64 data_count = u64_max;
    u64 load_value = u64_max;
};



class PatchTree{public:
    
    /**
     * @brief store the tree using a map
     * 
     */
    std::unordered_map<u64, PTNode> tree;

    /**
     * @brief key of leaf nodes in tree
     * 
     */
    std::unordered_set<u64> leaf_key;

    /**
     * @brief key of nodes that have only leafs as child
     * 
     */
    std::unordered_set<u64> parent_of_only_leaf_key;

    /**
     * @brief split a leaf node
     * 
     * @param id 
     */
    void split_node(u64 id);

    /**
     * @brief merge childs of idparent (id parent should have only leafs as childs)
     * 
     * @param idparent 
     */
    void merge_node_dm1(u64 idparent);

    /**
     * @brief make tree from a patch table
     * 
     * @param plist 
     * @param max_val_1axis 
     */
    void build_from_patchtable(std::vector<Patch> & plist, u64 max_val_1axis);

    /**
     * @brief update value in nodes (tree reduction) 
     * 
     * @param plist 
     * @param id_patch_to_global_idx 
     */
    void update_values_node(std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
    
    /**
     * @brief update values in leafs and parent_of_only_leaf_key only
     * 
     * @param plist 
     * @param id_patch_to_global_idx 
     */
    void partial_values_reduction(std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);


    

    /**
     * @brief Get list of nodes id to split
     * 
     * @param crit_load_split 
     * @return std::unordered_set<u64> 
     */
    inline std::unordered_set<u64> get_split_request(u64 crit_load_split){

        std::unordered_set<u64> rq;

        for(u64 a : leaf_key){
            if (tree[a].load_value > crit_load_split) {
                rq.insert(a);
            }
        }

        return rq;

    }

    /**
     * @brief Get list of nodes id to merge 
     * 
     * @param crit_load_merge 
     * @return std::unordered_set<u64> 
     */
    inline std::unordered_set<u64> get_merge_request(u64 crit_load_merge){
        std::unordered_set<u64> rq;

        for(u64 a : parent_of_only_leaf_key){
            if (tree[a].load_value < crit_load_merge) {
                rq.insert(a);
            }
        }

        return rq;
    }



    private:

    u64 next_id = 0;

    u64 insert_node(PTNode n);
    void remove_node(u64 id);

    void update_ptnode(PTNode & n,std::vector<Patch> & plist,std::unordered_map<u64,u64> id_patch_to_global_idx);
};