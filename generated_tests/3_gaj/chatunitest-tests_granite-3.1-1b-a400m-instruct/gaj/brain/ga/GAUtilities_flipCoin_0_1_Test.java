package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_flipCoin_0_1_Test {

    @Test
    void testFlipCoin() {
        // Should return true
        assertTrue(GAUtilities.flipCoin(0.5));
        // Should return false
        assertFalse(GAUtilities.flipCoin(0.5));
        // Should return false
        assertFalse(GAUtilities.flipCoin(1.0));
        // Should return false
        assertFalse(GAUtilities.flipCoin(-0.5));
        // Should return false
        assertFalse(GAUtilities.flipCoin(-1.0));
    }
}
