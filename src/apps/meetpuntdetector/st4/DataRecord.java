package apps.meetpuntdetector.st4;

public class DataRecord {

    private final int minute;
    private int intensityA, intensityB, velocityA, velocityB;

    public DataRecord(int minute) {
        this.minute = minute;
    }

    public void setIntensityA(int intensityA) {
        this.intensityA = intensityA;
    }

    public void setIntensityB(int intensityB) {
        this.intensityB = intensityB;
    }

    public void setVelocityA(int velocityA) {
        this.velocityA = velocityA;
    }

    public void setVelocityB(int velocityB) {
        this.velocityB = velocityB;
    }

    public String toString() {
        return String.format("Datarecord(minute: %d, %d/%d/%d/%d", minute, intensityA, intensityB, velocityA, velocityB);
    }
}
