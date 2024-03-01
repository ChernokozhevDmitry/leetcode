class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        for (std::vector<int>::iterator it = nums.begin() + 1; it != nums.end();){        
            if (*it == *(it - 1)) {
                nums.erase(it);
            }
            else {
                ++it;
            }
        }
        return nums.size();        
    }
};