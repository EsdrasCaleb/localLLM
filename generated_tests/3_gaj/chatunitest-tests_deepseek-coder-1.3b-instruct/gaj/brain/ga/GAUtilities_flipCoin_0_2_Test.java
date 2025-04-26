package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

class GAUtilities_flipCoin_0_2_Test {

    @Test
    void flipCoinTest() {
        // Test with a 50% chance
        boolean result = GAUtilities.flipCoin(0.5);
        assertTrue(result);
        // Test with a 100% chance
        result = GAUtilities.flipCoin(1.0);
        assertFalse(result);
        // Test with a 0% chance
        result = GAUtilities.flipCoin(0.0);
        assertFalse(result);
        // Test with a negative chance
        result = GAUtilities.flipCoin(-1.0);
        fail("Expected an IllegalArgumentException when providing a negative chance");
        // Test with a chance greater than 1.0
        result = GAUtilities.flipCoin(2.0);
        fail("Expected an IllegalArgumentException when providing a chance greater than 1.0");
    }
}
