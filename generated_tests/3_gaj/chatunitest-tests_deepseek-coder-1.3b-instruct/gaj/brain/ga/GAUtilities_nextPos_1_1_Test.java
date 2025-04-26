package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_nextPos_1_1_Test {

    @Test
    public void testNextPos() {
        int n = 5;
        for (int i = 0; i < 100; i++) {
            int result = GAUtilities.nextPos(n);
            assertTrue(result >= 0 && result <= n);
        }
    }
}
