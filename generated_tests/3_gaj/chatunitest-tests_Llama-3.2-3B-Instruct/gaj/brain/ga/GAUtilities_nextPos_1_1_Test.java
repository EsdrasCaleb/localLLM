package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class GAUtilities_nextPos_1_1_Test {

    @Test
    public void testNextPos_SmallRange() {
        int result = GAUtilities.nextPos(3);
        assertEquals(1, result);
    }

    @Test
    public void testNextPos_MidRange() {
        int result = GAUtilities.nextPos(10);
        assertEquals(4, result);
    }

    @Test
    public void testNextPos_LargeRange() {
        int result = GAUtilities.nextPos(100);
        assertEquals(50, result);
    }

    @Test
    public void testNextPos_RandomIndex() {
        int result = GAUtilities.nextPos(100);
        int expected = 1 + (int) (Math.random() * 100);
        assertEquals(expected, result);
    }

    @Test
    public void testNextPos_NegativeInput() {
        assertThrows(IllegalArgumentException.class, () -> GAUtilities.nextPos(-1));
    }

    @Test
    public void testNextPos_ZeroInput() {
        assertThrows(IllegalArgumentException.class, () -> GAUtilities.nextPos(0));
    }
}
