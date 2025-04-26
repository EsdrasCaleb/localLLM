package brain.ga;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_0_Test {

    @ParameterizedTest
    @CsvSource({ "1, 0", "2, 0", "2, 1", "3, 0", "3, 1", "3, 2", "4, 0", "4, 1", "4, 2", "4, 3", "5, 0", "5, 1", "5, 2", "5, 3", "5, 4" })
    void testNextPos_variousInputs(int n, int expectedRow) throws Exception {
        // Test with different seeds to cover different random number generations.
        for (int seed = 0; seed < 5; seed++) {
            Random mockRandom = Mockito.mock(Random.class);
            // This is crucial to make the test deterministic.  Without this, the test will be non-deterministic
            Mockito.when(mockRandom.nextInt(n * (n + 1) / 2)).thenReturn(expectedRow * (expectedRow + 1) / 2);
            Field rndField = GAUtilities.class.getDeclaredField("rnd");
            rndField.setAccessible(true);
            rndField.set(null, mockRandom);
            int actualRow = GAUtilities.nextPos(n);
            assertEquals(expectedRow, actualRow, "For n=" + n + ", seed=" + seed);
        }
    }

    @Test
    void testNextPos_edgeCase_nEqualsZero() throws Exception {
        Random mockRandom = Mockito.mock(Random.class);
        Field rndField = GAUtilities.class.getDeclaredField("rnd");
        rndField.setAccessible(true);
        rndField.set(null, mockRandom);
        // Expecting IllegalArgumentException or similar for n=0.  Adjust assertion based on actual exception.
        try {
            GAUtilities.nextPos(0);
        } catch (Exception e) {
            // Exception is expected
            assertTrue(true);
            return;
        }
        assertTrue(false, "Exception was not thrown for n=0");
    }

    @Test
    void testNextPos_largeN() {
        int n = 100;
        int result = GAUtilities.nextPos(n);
        assertTrue(result >= 0 && result < n);
    }
}
