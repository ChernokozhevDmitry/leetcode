class Solution {
public:
    int arrayPairSum(vector<int>& nums) {
        int res {0};
        std::sort(nums.begin(), nums.end());
        for(std::vector<int>::iterator i = nums.begin(); i != nums.end(); i += 2){
            res += std::min(*i, *(i + 1));
        }
        return res;        
    }
};