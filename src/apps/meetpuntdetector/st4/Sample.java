package apps.meetpuntdetector.st4;

public class Sample {
    private final int roadNumber, meetPuntIdx, meetPuntRange, minute, minutesInRange;
    private final char direction;

    public Sample(int roadNumber, char direction, int meetPuntIdx, int meetPuntRange, int minute, int minuteRange) {
        this.roadNumber = roadNumber;
        this.direction = direction;
        this.meetPuntIdx = meetPuntIdx;
        this.meetPuntRange = meetPuntRange;
        this.minute = minute;
        this.minutesInRange = minuteRange;
    }

    public int getRoadNumber() {
        return roadNumber;
    }

    public char getDirection() {
        return direction;
    }

    public boolean meetPuntInRange(int meetPunt) {
        return this.meetPuntIdx <= meetPunt && meetPunt < this.meetPuntIdx + meetPuntRange;
    }

    public boolean meetPuntAboveRange(int meetPunt) {
        return this.meetPuntIdx + meetPuntRange <= meetPunt;
    }

    public boolean minuteInRange(int minute) {
        return this.minute <= minute && minute < this.minute + minutesInRange;
    }

    public int getMinutesInRange() {
        return minutesInRange;
    }
}
