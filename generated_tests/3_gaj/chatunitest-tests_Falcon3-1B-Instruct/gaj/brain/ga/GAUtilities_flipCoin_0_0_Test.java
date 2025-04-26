package brain.ga;

import org.junit.Test;
import static org.junit.Assert.assertEquals;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.*;

public class GAUtilities_flipCoin_0_0_Test {

    @Test
    public void testFlipCoinWithSpecificProbability() {
        // Arrange
        GAUtilities gaus = new GAUtilities();
        // Desired probability of heads
        double probability = 0.8;
        // Act
        boolean result = gaus.flipCoin(probability);
        // Assert
        // Expected result: true
        assertEquals(true, result);
    }
}
