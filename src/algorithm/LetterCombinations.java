package algorithm;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LetterCombinations {

    List<String> result = new ArrayList<>();

    /**
     * 17. 电话号码的字母组合
     * @param digits
     * @return
     */
    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) return result;
        Map<Character, String> map = new HashMap<>();
        map.put('2', "abc");
        map.put('3', "def");
        map.put('4', "ghi");
        map.put('5', "jkl");
        map.put('6', "mno");
        map.put('7', "pqrs");
        map.put('8', "tuv");
        map.put('9', "wxyz");
        StringBuilder path = new StringBuilder();
        dfs(digits, 0, path, map);
        return result;
    }

    private void dfs(String digits, int index, StringBuilder path, Map<Character, String> map){
        // index 越界返回
        if (index == digits.length()){
            result.add(path.toString());
            return;
        }
        // 取其对应的字母
        char c = digits.charAt(index);
        String s = map.get(c);
        // 每遍历一个字母查找下一个元素的字母
        for (int j = 0; j < s.length(); j++) {
            path.append(s.charAt(j));
            dfs(digits, index + 1, path, map);
            path.deleteCharAt(path.length() - 1);
        }
    }

    public static void main(String[] args) {
        LetterCombinations letterCombinations = new LetterCombinations();
        System.out.println(letterCombinations.letterCombinations("234"));
    }
}
