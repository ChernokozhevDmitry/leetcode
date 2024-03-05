class Solution {
public:
    int thirdMax(vector<int>& nums) {
        std::sort(nums.rbegin(), nums.rend());
        int max1 = 0, max2 = 0, max3 = 0, count = 0;
        if (!nums.empty()) {
            max1 = nums[0];
            count = 1;
        }
        for(int i = 1; i < nums.size(); ++i){
            if ((nums[i] < max1) && (count == 1)){
                ++count;
                max2 = nums[i];
                continue;
            }
            if ((nums[i] < max2) && (count == 2)){
                ++count;
                max3 = nums[i];
                exit;
            }
        }        
        switch(count){
            case 1: return max1;
            case 2: return max1;
            case 3: return max3;
            default: return 0;
        }
    }
};