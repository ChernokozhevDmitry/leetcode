class Solution {
public:
    int heightChecker(vector<int>& heights) {
        std::vector<int> expected = heights;
        int val = 0;
        std::sort(expected.begin(), expected.end());
        for(std::vector<int>::iterator it1 = expected.begin(), it2 = heights.begin();
                                                 it1 != expected.end(); ++it1, ++it2){
            if (*it1 != *it2){
                ++val;
            }
        }        
        return val;
    }
};