package brain.ga;

import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class GAUtilities_nextPos_1_1_Test {

    @InjectMocks
    private GAUtilities gaUtilities;

    @Mock
    private Random rnd;

    @BeforeEach
    public void setUp() throws Exception {
        Field rndField = GAUtilities.class.getDeclaredField("rnd");
        rndField.setAccessible(true);
        rndField.set(gaUtilities, rnd);
    }

    @Test
    public void testNextPos() {
        when(rnd.nextInt(anyInt())).thenReturn(1);
        int result = gaUtilities.nextPos(1);
        assertEquals(1, result);
    }

    @Test
    public void testNextPosBoundary() throws Exception {
        int n = 1;
        when(rnd.nextInt(1)).thenReturn(0);
        int result = gaUtilities.nextPos(n);
        assertEquals(1, result);
    }

    @Test
    public void testNextPosLargeInput() throws Exception {
        int n = 1000;
        int expectedRange = n * (n + 1) / 2;
        when(rnd.nextInt(expectedRange)).thenReturn(expectedRange / 2);
        int result = gaUtilities.nextPos(n);
        assertTrue(result >= 1 && result <= n);
    }
}
