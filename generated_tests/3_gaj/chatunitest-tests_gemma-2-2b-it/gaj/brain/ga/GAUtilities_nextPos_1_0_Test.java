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
        assertEquals(GAUtilities.nextPos(5), 4);
        assertEquals(GAUtilities.nextPos(10), 9);
        assertEquals(GAUtilities.nextPos(20), 19);
        assertEquals(GAUtilities.nextPos(0), 0);
        assertEquals(GAUtilities.nextPos(1), 0);
        assertEquals(GAUtilities.nextPos(100), 99);
    }
}
