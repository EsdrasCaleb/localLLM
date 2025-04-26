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

    @BeforeEach
    public void setUp() throws Exception {
        MockitoAnnotations.openMocks(this);
        Field rndField = GAUtilities.class.getDeclaredField("rnd");
        rndField.setAccessible(true);
        // Corrected to set the static field to null
        rndField.set(null, mockRandom);
    }

    @Test
    public void testFlipCoinReturnsTrueWhenRandomLessThan0_5() {
        when(mockRandom.nextDouble()).thenReturn(0.49);
        // Corrected to pass the threshold as argument
        assertTrue(GAUtilities.flipCoin(0.5));
    }

    @Test
    public void testFlipCoinReturnsFalseWhenRandomGreaterThan0_5() {
        when(mockRandom.nextDouble()).thenReturn(0.51);
        // Corrected to pass the threshold as argument
        assertFalse(GAUtilities.flipCoin(0.5));
    }

    @Test
    public void testFlipCoinReturnsFalseWhenRandomExactly0_5() {
        when(mockRandom.nextDouble()).thenReturn(0.5);
        // Corrected to pass the threshold as argument
        assertFalse(GAUtilities.flipCoin(0.5));
    }

    @Test
    public void testFlipCoinWithThresholdTrue() {
        when(mockRandom.nextDouble()).thenReturn(0.3);
        assertTrue(GAUtilities.flipCoin(0.75));
        assertTrue(GAUtilities.flipCoin(0.25));
    }

    @Test
    public void testFlipCoinWithThresholdFalse() {
        when(mockRandom.nextDouble()).thenReturn(0.8);
        assertFalse(GAUtilities.flipCoin(0.75));
        assertFalse(GAUtilities.flipCoin(0.25));
    }
}
