package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_0_Test {

    @Test
    void testNextPos() {
        // Given
        int n = 10;
        int expected = 3;
        // When
        int actual = GAUtilities.nextPos(n);
        // Then
        assertEquals(expected, actual);
    }

    @Test
    void testNextPosNegative() {
        // Given
        int n = 10;
        int expected = -1;
        // When
        int actual = GAUtilities.nextPos(n);
        // Then
        assertEquals(expected, actual);
    }

    @Test
    void testNextPosPositive() {
        // Given
        int n = 10;
        int expected = 5;
        // When
        int actual = GAUtilities.nextPos(n);
        // Then
        assertEquals(expected, actual);
    }
}
