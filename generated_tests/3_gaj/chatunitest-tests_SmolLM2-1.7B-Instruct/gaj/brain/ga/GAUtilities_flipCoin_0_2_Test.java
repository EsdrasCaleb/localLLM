package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_flipCoin_0_2_Test {

    // Test method
    @Test
    public void testFlippingCoin() {
        assertFalse(GAUtilities.flipCoin(0.5));
        assertTrue(GAUtilities.flipCoin(0.7));
    }
}
