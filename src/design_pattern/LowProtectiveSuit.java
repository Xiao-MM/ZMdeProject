package design_pattern;

public class LowProtectiveSuit implements ProtectiveSuit{
    @Override
    public void wear() {
        System.out.println("低端防护服");
    }
}
