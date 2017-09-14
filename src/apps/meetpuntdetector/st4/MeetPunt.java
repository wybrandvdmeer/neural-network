package apps.meetpuntdetector.st4;

public class MeetPunt {

    private final long id;
    private int distanceToDS, noOfLanesA, noOfLanesB, osType, gp, laneData;
    private long meterPosition, showMap, blockMap;
    private String roadName;

    public MeetPunt(long id) {
        this.id = id;
    }

    public void setRoadName(String roadName) {
        this.roadName = roadName.trim();
    }

    public void setMeterPosition(long meterPosition) {
        this.meterPosition = meterPosition;
    }

    public void setDistanceToDS(int distanceToDS) {
        this.distanceToDS = distanceToDS;
    }

    public void setNoOfLanesA(int noOfLanesA) {
        this.noOfLanesA = noOfLanesA;
    }

    public void setNoOfLanesB(int noOfLanesB) {
        this.noOfLanesB = noOfLanesB;
    }

    public void setOSType(int osType) {
        this.osType = osType;
    }

    public void setGP(int gp) {
        this.gp = gp;
    }

    public void setLaneData(int laneData) {
        this.laneData = laneData;
    }

    public void setShowMap(long showMap) {
        this.showMap = showMap;
    }

    public void setBlockMap(long blockMap) {
        this.blockMap = blockMap;
    }

    public String toString() {
        return String.format("Meetpunt(%d, %s)", id, roadName);
    }

    public boolean isMainRoad() {
        return roadName.charAt(roadName.length() - 2) == ' ';
    }

    public char getDirection() {
        return roadName.charAt(roadName.length() - 1);
    }

    public int getRoadNumber() {

        String s="";

        for(char c : roadName.toCharArray()) {
            if(Character.isDigit(c)) {
                s += c;
            }
        }

        return Integer.parseInt(s);
    }

    public boolean isSuspect() {
        return !Character.isDigit(roadName.charAt(0));
    }
}
