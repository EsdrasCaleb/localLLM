package brain.ga;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.util.*;

@ExtendWith(MockitoExtension.class)
public class GAUtilities_nextPos_1_0_Test {

    @Mock
    private Random random;

    @InjectMocks
    private GAUtilities focal;

    @Test
    public void testNextPos() {
        // Arrange
        int n = 10;
        int index = focal.nextPos(n);
        // Act
        int actual = focal.nextPos(n);
        // Assert
        assertEquals(n - 1, actual);
    }
}
