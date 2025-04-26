package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class GAUtilities_nextPos_1_2_Test {

    @Mock
    private Random random;

    @Captor
    private ArgumentCaptor<Integer> captor;

    @Test
    public void testNextPos() {
        try (MockedStatic<Random> mockedStatic = mockStatic(Random.class)) {
            mockedStatic.when(random::nextInt).thenReturn(10, 20, 30);
            assertEquals(3, GAUtilities.nextPos(5));
            assertEquals(4, GAUtilities.nextPos(6));
            assertEquals(5, GAUtilities.nextPos(10));
        }
    }
}
