class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        std::vector<int> res;
        for(std::vector<int>::iterator it = numbers.begin(); it != numbers.end(); ++it){
            for(std::vector<int>::iterator jt = it + 1; jt != numbers.end(); ++jt){
                if (target == (*it + *jt)){
                    res.push_back(std::distance( numbers.begin(), it ) + 1);
                    res.push_back(std::distance( numbers.begin(), jt ) + 1);
                    return res;
                }
            }
        }        
        return res;
    }
};
