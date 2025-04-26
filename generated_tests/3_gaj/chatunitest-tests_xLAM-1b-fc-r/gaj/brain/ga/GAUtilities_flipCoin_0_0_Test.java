package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_flipCoin_0_0_Test {

    @Test
    void flipCoinTest() {
        // Test with a probability of 0.5
        boolean result = GAUtilities.flipCoin(0.5);
        assertTrue(result);
        // Test with a probability of 0.0
        result = GAUtilities.flipCoin(0.0);
        assertFalse(result);
        // Test with a probability of 1.0
        result = GAUtilities.flipCoin(1.0);
        assertTrue(result);
        // Test with a probability of 0.0
        result = GAUtilities.flipCoin(0.0);
        assertFalse(result);
        // Test with a probability of 0.0
        result = GAUtilities.flipCoin(0.0);
        assertFalse(result);
        // Test with a probability of 1.0
        result = GAUtilities.flipCoin(1.0);
        assertTrue(result);
    }
}
