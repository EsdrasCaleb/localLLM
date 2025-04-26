package brain.ga;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_flipCoin_0_0_Test {

    @Test
    public void testFlipCoin() {
        GAUtilities util = new GAUtilities();
        // Test with probability of 0.5
        assertTrue(util.flipCoin(0.5));
        // Test with probability of 0.5
        assertFalse(util.flipCoin(0.5));
    }
}
