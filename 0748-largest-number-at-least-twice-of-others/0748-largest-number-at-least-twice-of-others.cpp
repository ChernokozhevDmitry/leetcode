class Solution {
public:
    int dominantIndex(vector<int>& nums) {
        int nummax = *std::max_element(nums.begin(), nums.end());
        auto nummax_it = std::max_element(nums.begin(), nums.end());
        std::cout << nummax << ' ' << nummax/2 << std::endl;
        for(auto it = nums.begin(); it != nums.end(); ++it){
            if (it != nummax_it) {
                if (*it > nummax/2) {
                    std::cout << *it << ' ' << nummax << std::endl;
                    return -1;
                }
            }
        }    
        return std::distance(nums.begin(), nummax_it);
    }
};