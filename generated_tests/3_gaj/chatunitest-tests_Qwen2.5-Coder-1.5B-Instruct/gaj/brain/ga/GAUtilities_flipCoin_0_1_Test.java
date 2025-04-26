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
    public void testFlipCoin() throws Exception {
        // Create an instance of GAUtilities
        GAUtilities gaUtilities = new GAUtilities();
        // Test cases
        double[] probabilities = { 0.2, 0.3, 0.4, 0.5 };
        boolean[] expectedResults = { false, false, true, true };
        for (int i = 0; i < probabilities.length; i++) {
            // Invoke the flipCoin method
            boolean result = gaUtilities.flipCoin(probabilities[i]);
            // Assert the result
            assertEquals(expectedResults[i], result);
        }
    }

    @Test
    public void testFlipCoinNegativeProbability() throws Exception {
        // Create an instance of GAUtilities
        GAUtilities gaUtilities = new GAUtilities();
        // Test case
        double negativeProbability = -0.1;
        // Invoke the flipCoin method
        boolean result = gaUtilities.flipCoin(negativeProbability);
        // Assert the result
        assertEquals(false, result);
    }
}
