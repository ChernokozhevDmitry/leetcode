class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        for(int& i: nums){
            i = pow(i, 2);
        }
        std::sort(nums.begin(), nums.end());
        return nums;
    }
};