package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_0_Test {

    private static Random rnd = new Random();

    @Test
    public void testNextPos() {
        int n = 10;
        for (int i = 0; i < 10; i++) {
            int expected = rnd.nextInt(n * (n + 1) / 2) + 1;
            int actual = GAUtilities.nextPos(n);
            assertEquals(expected, actual, "Test failed for n=" + n);
        }
    }
}
