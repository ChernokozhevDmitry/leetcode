class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int chain_max = 0, chain_cur = 0;
        for(auto i: nums){
            std::cout << "i = "<< i << std::endl;
            if (i == 1) {
                ++chain_cur;
                std::cout << "chain_cur = "<< chain_cur << std::endl;
            }
            else {
                if (chain_cur > chain_max) {
                    chain_max = chain_cur;
                }
                chain_cur = 0;
            }
        }
        if (chain_cur > chain_max) {
            chain_max = chain_cur;
        }
        return chain_max;
    }
};