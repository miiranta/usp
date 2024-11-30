package model;

/**
 * The MachineData models are represented in this enumerator
 * each one is associated with a code
 */
public enum MachineDataModel {
    BLUE {
        @Override
        public String toString() {
            return "1";
        }
    },
    GREEN {
        @Override
        public String toString() {
            return "2";
        }
    },
    YELLOW {
        @Override
        public String toString() {
            return "3";
        }
    },
    NEW {
        @Override
        public String toString() {
            return "4";
        }
    },
    UNKNOWN {
        @Override
        public String toString() {
            return "0";
        }
    }
}
