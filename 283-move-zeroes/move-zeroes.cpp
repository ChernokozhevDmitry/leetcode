class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int count = 0;
        for(std::vector<int>::iterator it=nums.begin(); it != nums.end();){
            if (*it == 0){
                it = nums.erase(it);
                ++count;
            }
            else {
                ++it;
            }
        }
        for(int i = 0; i < count; ++i){
            nums.push_back(0);
        }
    }
};