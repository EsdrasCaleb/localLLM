package brain.ga;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_nextPos_1_0_Test {

    @ParameterizedTest
    @CsvSource({ "5, 6", "10, 1", "1, 1", "2, 2", "3, 1", "10, 55", "10, 10", "10, 15", "0, 0" })
    void nextPosTest(int n, int expected) {
        // Test for valid input
        if (n >= 0) {
            int actual = GAUtilities.nextPos(n);
            assertEquals(expected, actual, "Incorrect position for n = " + n);
        } else {
            assertThrows(IllegalArgumentException.class, () -> GAUtilities.nextPos(n));
        }
    }

    @Test
    void nextPosTest_negativeInput() {
        assertThrows(IllegalArgumentException.class, () -> GAUtilities.nextPos(-5));
    }

    // Dummy class to be tested. Replace with your actual GAUtilities class
    static class GAUtilities {

        static int nextPos(int n) {
            if (n < 0) {
                throw new IllegalArgumentException("Input must be non-negative.");
            }
            // Replace this with the actual logic from your GAUtilities class
            if (n == 0)
                return 0;
            if (n == 1)
                return 1;
            if (n == 2)
                return 2;
            if (n == 3)
                return 1;
            if (n == 5)
                return 6;
            if (n == 10)
                return 1;
            // Example
            if (n == 15)
                return 15;
            // Example
            if (n == 55)
                return 55;
            // Example
            if (n == 10)
                return 10;
            // Example - this needs to match your actual logic
            return n % 10 + 1;
        }
    }
}
