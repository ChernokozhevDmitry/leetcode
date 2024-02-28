class Solution {
public:
    int findNumbers(vector<int>& nums) {
        int count = 0;
        for(auto i: nums){
        if (static_cast<int>((log10(i))+1)%2 == 0) {
                ++count;
            }
        }
        return count;
        
    }
};