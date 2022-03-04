package algorithm;


import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class RestoreIpAddresses {

    List<String> result;
    LinkedList<String> path;

    /**
     * 93. 复原 IP 地址
     * @param s
     * @return
     */
    public List<String> restoreIpAddresses(String s) {
        result = new ArrayList<>();
        path = new LinkedList<>();
        dfs(s, 0, 0, s.length());
        return result;
    }

    private void dfs(String s, int begin, int depth, int length){
        // 递归深度到达4时就可以返回了
        if (depth == 4) {
            // 符合条件添加
            StringBuilder sb = new StringBuilder();
            for (String value : path) sb.append(value).append(".");
            sb.deleteCharAt(sb.length() - 1);
            result.add(sb.toString());
            return;
        }
        // 后续长度超了需要的长度时返回
        if((4 - depth) * 3 < length - begin) return;
        // i 这里为截取位置
        for (int i = begin + 1; i <= length; i++) {
            // 到最后一层了直接截取最后一位
            if (depth == 3) i = length;
            // 截取 begin -> i 之间的字符串
            String num = s.substring(begin, i);
            // 0 打头的数据不对直接返回
            if (num.length() > 1 && num.charAt(0) == '0') return;
            // 数据违规，大于255
            if (Integer.parseInt(num) > 255) return;
            path.addLast(num);
            dfs(s, i, depth + 1, length);
            path.removeLast();
        }
    }

    public static void main(String[] args) {
        RestoreIpAddresses restoreIpAddresses = new RestoreIpAddresses();
        System.out.println(restoreIpAddresses.restoreIpAddresses("0000"));
    }
}
