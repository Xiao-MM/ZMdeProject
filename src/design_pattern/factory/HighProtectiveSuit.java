package design_pattern.factory;

public class HighProtectiveSuit implements ProtectiveSuit{
    @Override
    public void wear() {
        System.out.println("高端防护服");
    }
}
