package design_pattern;

/**
 * 枚举实现
 */
public enum SingletonEnum {

    INSTANCE;

    private String objectName;

    public String getObjectName() {
        return objectName;
    }

    public void setObjectName(String objectName) {
        this.objectName = objectName;
    }

    public static void main(String[] args) {
        SingletonEnum firstObject = SingletonEnum.INSTANCE;
        firstObject.setObjectName("zs");
        System.out.println(firstObject.getObjectName());
        SingletonEnum secondObject = SingletonEnum.INSTANCE;
        secondObject.setObjectName("ls");
        System.out.println(firstObject.getObjectName());
        System.out.println(secondObject.getObjectName());

        // 反射获取实例测试
        try {
            SingletonEnum[] enumConstants = SingletonEnum.class.getEnumConstants();
            for (SingletonEnum enumConstant : enumConstants) {
                System.out.println(enumConstant.getObjectName());
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
