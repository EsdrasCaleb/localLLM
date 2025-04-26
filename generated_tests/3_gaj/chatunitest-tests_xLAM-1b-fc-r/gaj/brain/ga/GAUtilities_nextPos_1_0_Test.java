package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_nextPos_1_0_Test {

    @Test
    void nextPosTest() {
        // Test with a small value
        int n = 5;
        int result = GAUtilities.nextPos(n);
        assertTrue(result >= 0 && result < n);
        // Test with a large value
        n = 10000;
        result = GAUtilities.nextPos(n);
        assertTrue(result >= 0 && result < n);
        // Test with a negative value
        n = -10;
        result = GAUtilities.nextPos(n);
        assertTrue(result >= 0 && result < n);
        // Test with 0
        n = 0;
        result = GAUtilities.nextPos(n);
        assertTrue(result >= 0 && result < n);
    }
}
