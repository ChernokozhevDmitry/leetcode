class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res = nums.size() + 1, sum = 0, j = 0;
        for(int i = 0; i != nums.size(); ++i){
            sum += nums[i];
            while(sum >= target){
                res = std::min(res, i - j + 1);
                sum = sum - nums[j];
                ++j;
            }
        }
        return res == (nums.size() + 1) ? 0 : res ;        
    }
};