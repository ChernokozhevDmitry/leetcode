class Solution {
public:
    int findMaxConsecutiveOnes(vector<int>& nums) {
        int chain_max = 0, chain_cur = 0;
        for(auto i: nums){
            if (i == 1) {
                ++chain_cur;
                chain_max = max(chain_max, chain_cur);
            }
            else {
                chain_cur = 0;
            }
        }
        return chain_max;
    }
};