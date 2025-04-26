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
public class GAUtilities_flipCoin_0_0_Test {

    @Mock
    private Random mockRandom;

    @InjectMocks
    private GAUtilities gaUtilities;

    @BeforeEach
    public void setUp() throws Exception {
        // Use reflection to set the private static field `rnd` to the mockRandom
        Field field = GAUtilities.class.getDeclaredField("rnd");
        field.setAccessible(true);
        field.set(null, mockRandom);
    }

    @Test
    public void testFlipCoin() {
        // Arrange
        when(mockRandom.nextDouble()).thenReturn(0.49);
        // Act
        boolean result = gaUtilities.flipCoin(0.5);
        // Assert
        assertFalse(result);
    }

    @Test
    public void testFlipCoin_EdgeCase() {
        // Arrange
        when(mockRandom.nextDouble()).thenReturn(0.5);
        // Act
        boolean result = gaUtilities.flipCoin(0.5);
        // Assert
        assertTrue(result);
    }

    @Test
    public void testFlipCoinTrue() {
        when(mockRandom.nextBoolean()).thenReturn(true);
        assertTrue(GAUtilities.flipCoin(0.5));
    }

    @Test
    public void testFlipCoinFalse() {
        when(mockRandom.nextBoolean()).thenReturn(false);
        assertFalse(GAUtilities.flipCoin(0.5));
    }
}
