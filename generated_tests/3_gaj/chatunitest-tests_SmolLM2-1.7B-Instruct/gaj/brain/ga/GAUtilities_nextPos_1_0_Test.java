package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
class GAUtilities_nextPos_1_0_Test {

    @Mock
    private Random rnd;

    @Test
    public void testNextPos() {
        when(rnd.nextInt(100)).thenReturn(20);
        assertEquals(20, GAUtilities.nextPos(100));
    }
}
