package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_flipCoin_0_0_Test {

    @BeforeEach
    public void setUp() {
        // Initialize the Random object
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testFlipCoin() {
        // Create a mock instance of Random
        Random mockRandom = Mockito.mock(Random.class);
        // Set up the expected behavior of the flipCoin method
        // 50% chance of flipping a coin
        when(mockRandom.nextDouble()).thenReturn(0.5);
        // Call the flipCoin method with a probability of 0.5
        boolean result = GAUtilities.flipCoin(0.5);
        // Verify that the result is correct
        assertTrue(result);
    }
}
