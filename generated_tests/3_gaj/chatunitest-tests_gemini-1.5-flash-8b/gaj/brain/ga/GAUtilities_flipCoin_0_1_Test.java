package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_flipCoin_0_1_Test {

    @Test
    void testFlipCoin() {
        // Test with various probabilities, including edge cases
        // Test case 1: Probability > 0.5
        assertTrue(GAUtilities.flipCoin(0.8));
        assertFalse(GAUtilities.flipCoin(0.8));
        // Test case 2: Probability = 0.5
        assertTrue(GAUtilities.flipCoin(0.5));
        assertFalse(GAUtilities.flipCoin(0.5));
        // Test case 3: Probability < 0.5
        assertTrue(GAUtilities.flipCoin(0.2));
        assertFalse(GAUtilities.flipCoin(0.2));
        // Test case 4: Probability = 0.0
        assertTrue(GAUtilities.flipCoin(0.0));
        assertFalse(GAUtilities.flipCoin(0.0));
        // Test case 5: Probability = 1.0
        assertTrue(GAUtilities.flipCoin(1.0));
        assertFalse(GAUtilities.flipCoin(1.0));
    }
}
