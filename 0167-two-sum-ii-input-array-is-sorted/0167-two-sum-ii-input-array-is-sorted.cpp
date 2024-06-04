class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        std::vector<int> res;
        std::vector<int>::iterator it = numbers.begin(), jt = numbers.end() - 1;
        while(it < jt){
            if (target == (*it + *jt)){
                    res.push_back(std::distance( numbers.begin(), it ) + 1);
                    res.push_back(std::distance( numbers.begin(), jt ) + 1);
                    return res;
            }
            else if (target > (*it + *jt)) {
                ++it;
            } else {
                --jt;
            }
        } 
        return res;
    }
};